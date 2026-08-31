"""Fused Gated MLP Triton kernel implementation.

This module implements a fused kernel for Gated MLP (used in LLaMA, Mistral, etc.)
that combines gate projection, up projection, and activation into a single kernel.

Mathematical formula:
output = activation(gate_proj(x)) * up_proj(x)

Where activation is either SiLU (x * sigmoid(x)) or GELU.
"""

import torch
import triton
import triton.language as tl

from trifuse.validation import (
    ACTIVATION_GELU,
    ACTIVATION_SILU,
    VALID_ACTIVATIONS,
    validate_gated_mlp_inputs,
)

# torch dtype -> Triton dtype 映射（Triton 3.x 不再接受 torch.dtype 作 constexpr）
_DTYPE_MAP = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


@triton.jit
def silu(x):
    """SiLU (Swish) activation: x * sigmoid(x)"""
    return x * tl.sigmoid(x)


@triton.jit
def gelu(x):
    """GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))"""
    return x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))


@triton.jit
def fused_gated_mlp_kernel(
    x_ptr,
    gate_weight_ptr,
    up_weight_ptr,
    output_ptr,
    # Strides for x [batch, seq_len, hidden_dim]
    stride_x_batch,
    stride_x_seq,
    stride_x_hidden,
    # Strides for weights [intermediate_dim, hidden_dim]
    stride_gw_inter,
    stride_gw_hidden,
    stride_uw_inter,
    stride_uw_hidden,
    # Strides for output [batch, seq_len, intermediate_dim]
    stride_out_batch,
    stride_out_seq,
    stride_out_inter,
    # Dimensions
    batch_size,
    seq_len,
    hidden_dim,
    intermediate_dim,
    # Activation type: 0=SiLU, 1=GELU
    activation_type: tl.constexpr,
    out_dtype: tl.constexpr,
    # fp32 输入禁 TF32（"ieee"）；fp16/bf16 输入不受影响（"tf32" 占位）
    input_precision: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused Gated MLP kernel.

    Computes: output = activation(gate_proj(x)) * up_proj(x)
    Each program computes a BLOCK_M x BLOCK_N tile of the output.
    """
    # Program ID
    pid = tl.program_id(0)

    # Compute which tile this program handles
    num_tiles_m = tl.cdiv(batch_size * seq_len, BLOCK_M)
    num_tiles_n = tl.cdiv(intermediate_dim, BLOCK_N)

    tile_m = pid // num_tiles_n
    tile_n = pid % num_tiles_n

    # Compute row and column ranges for this tile
    row_start = tile_m * BLOCK_M
    col_start = tile_n * BLOCK_N

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    # Compute batch and seq indices from row index
    batch_indices = rows // seq_len
    seq_indices = rows % seq_len

    # Initialize accumulators for gate and up projections
    gate_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    up_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Compute matrix multiplication in blocks
    for k_start in range(0, hidden_dim, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < hidden_dim

        # Load x block [BLOCK_M, BLOCK_K]
        x_ptrs = (
            x_ptr
            + batch_indices[:, None] * stride_x_batch
            + seq_indices[:, None] * stride_x_seq
            + k_range[None, :] * stride_x_hidden
        )
        row_mask = rows[:, None] < (batch_size * seq_len)
        x_mask = row_mask & k_mask[None, :]
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load gate weight block [BLOCK_K, BLOCK_N]
        # Need transpose: weight is [intermediate_dim, hidden_dim] but we need [hidden_dim, intermediate_dim]
        gw_ptrs = (
            gate_weight_ptr + k_range[:, None] * stride_gw_hidden + cols[None, :] * stride_gw_inter
        )
        col_mask = cols[None, :] < intermediate_dim
        gw_mask = k_mask[:, None] & col_mask
        gw_block = tl.load(gw_ptrs, mask=gw_mask, other=0.0)

        # Load up weight block [BLOCK_K, BLOCK_N]
        uw_ptrs = (
            up_weight_ptr + k_range[:, None] * stride_uw_hidden + cols[None, :] * stride_uw_inter
        )
        uw_block = tl.load(uw_ptrs, mask=gw_mask, other=0.0)

        # Accumulate matrix products（原生 dtype 送入 tl.dot）
        #
        # 这里刻意不把 fp16/bf16 输入提升到 fp32 再 dot：fp16/bf16 原生 dot 直接
        # 走张量核心并在 fp32 里累积，又快又准；若强转 fp32 再 dot，则 triton 只能
        # 用 CUDA 核心/AI 核心且精度路线更慢（实测稳态延迟慢 ~8 倍）。
        #
        gate_acc = tl.dot(
            x_block,
            gw_block,
            gate_acc,
            input_precision=input_precision,
            out_dtype=tl.float32,
        )
        up_acc = tl.dot(
            x_block,
            uw_block,
            up_acc,
            input_precision=input_precision,
            out_dtype=tl.float32,
        )

    # Standard SwiGLU/GeGLU applies the non-linearity to the gate projection.
    if activation_type == 0:
        gate_activated = silu(gate_acc)
    else:
        gate_activated = gelu(gate_acc)

    output = gate_activated * up_acc

    # Store output
    out_ptrs = (
        output_ptr
        + batch_indices[:, None] * stride_out_batch
        + seq_indices[:, None] * stride_out_seq
        + cols[None, :] * stride_out_inter
    )
    out_mask = (rows[:, None] < (batch_size * seq_len)) & (cols[None, :] < intermediate_dim)

    # Convert to output dtype
    tl.store(out_ptrs, output.to(out_dtype), mask=out_mask)


def _require_pow2(value: int, name: str) -> None:
    """BLOCK_* 必须是 2 的幂，否则 tl.arange 会在编译期崩溃。"""
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or (value & (value - 1)) != 0
    ):
        raise ValueError(f"{name} must be a power of two >= 1, got {value}")


def fused_gated_mlp(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    activation: str = "silu",
    BLOCK_M: int = 32,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Apply fused Gated MLP transformation.

    Computes: output = activation(gate_proj(x)) * up_proj(x), matching
    standard SwiGLU/GeGLU.

    This fused implementation reduces memory bandwidth by computing both
    projections and the activation in a single kernel.

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim]
        gate_weight: Gate projection weight of shape [intermediate_dim, hidden_dim]
        up_weight: Up projection weight of shape [intermediate_dim, hidden_dim]
        activation: Activation function - "silu" (default) or "gelu"

    Returns:
        Output tensor of shape [batch, seq_len, intermediate_dim]

    Raises:
        DeviceError: If CUDA is not available
        ShapeMismatchError: If tensor shapes are incompatible
        UnsupportedDtypeError: If tensor dtypes are unsupported
        ValueError: If activation is not supported

    Example:
        >>> x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
        >>> gate_w = torch.randn(11264, 4096, device='cuda', dtype=torch.float16)
        >>> up_w = torch.randn(11264, 4096, device='cuda', dtype=torch.float16)
        >>> output = fused_gated_mlp(x, gate_w, up_w, activation="silu")

    Note:
        All tensors must be on CUDA device and contiguous.
    """
    # Validate inputs
    batch_size, seq_len, hidden_dim, intermediate_dim = validate_gated_mlp_inputs(
        x, gate_weight, up_weight, activation
    )

    # 权重与激活 dtype 不一致时（如 module 参数为 fp32、输入为 fp16），把权重
    # cast 到 x 的 dtype。kernel 的 tl.dot 要求两端 dtype 一致（原生 dtype 路径），
    # 且 fp16/bf16 输入下 upcast 权重会丢掉张量核心的快路径。
    if gate_weight.dtype != x.dtype:
        gate_weight = gate_weight.to(x.dtype)
    if up_weight.dtype != x.dtype:
        up_weight = up_weight.to(x.dtype)

    # Handle empty tensors
    if batch_size == 0 or seq_len == 0 or hidden_dim == 0 or intermediate_dim == 0:
        return torch.empty(batch_size, seq_len, intermediate_dim, dtype=x.dtype, device=x.device)

    # Allocate output
    output = torch.empty(batch_size, seq_len, intermediate_dim, dtype=x.dtype, device=x.device)

    # Validate and determine activation type
    if activation not in VALID_ACTIVATIONS:
        raise ValueError(
            f"activation must be '{ACTIVATION_SILU}' or '{ACTIVATION_GELU}', got '{activation}'"
        )
    activation_type = 0 if activation == ACTIVATION_SILU else 1
    # fp32 输入必须禁 TF32（否则精度损失 ~2 个数量级，见 sgemm.py 同一约定）；
    # fp16/bf16 输入不受该参数影响。
    input_precision = "ieee" if x.dtype == torch.float32 else "tf32"

    # Block sizes (power-of-2 required by tl.arange; exposed for auto-tuning)
    _require_pow2(BLOCK_M, "BLOCK_M")
    _require_pow2(BLOCK_N, "BLOCK_N")
    _require_pow2(BLOCK_K, "BLOCK_K")

    # Grid size
    num_tiles_m = triton.cdiv(batch_size * seq_len, BLOCK_M)
    num_tiles_n = triton.cdiv(intermediate_dim, BLOCK_N)
    grid = (num_tiles_m * num_tiles_n,)

    # Launch kernel
    fused_gated_mlp_kernel[grid](
        x,
        gate_weight,
        up_weight,
        output,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        gate_weight.stride(0),
        gate_weight.stride(1),
        up_weight.stride(0),
        up_weight.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        batch_size,
        seq_len,
        hidden_dim,
        intermediate_dim,
        activation_type=activation_type,
        out_dtype=_DTYPE_MAP[x.dtype],
        input_precision=input_precision,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output


class FusedGatedMLP(torch.nn.Module):
    """PyTorch module wrapper for fused Gated MLP.

    This module provides a convenient interface for using the fused kernel
    in PyTorch models.

    Args:
        hidden_dim: Input hidden dimension
        intermediate_dim: Intermediate (FFN) dimension
        activation: Activation function ("silu" or "gelu")
    """

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        activation: str = "silu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activation

        # Weight parameters
        self.gate_weight = torch.nn.Parameter(torch.randn(intermediate_dim, hidden_dim) * 0.02)
        self.up_weight = torch.nn.Parameter(torch.randn(intermediate_dim, hidden_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fused Gated MLP.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]

        Returns:
            Output tensor [batch, seq_len, intermediate_dim]
        """
        return fused_gated_mlp(x, self.gate_weight, self.up_weight, self.activation)
