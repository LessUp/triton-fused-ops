"""Fused RMSNorm + RoPE Triton kernel implementation.

This module implements a fused kernel that combines RMSNorm and Rotary Position Embedding (RoPE)
into a single GPU kernel, reducing HBM access from 3 to 1.

Mathematical formulas:
- RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
- RoPE: x_rope = x * cos(theta) + rotate_half(x) * sin(theta)
"""

import torch
import triton
import triton.language as tl

from trifuse.validation import (
    validate_eps,
    validate_head_dim,
    validate_positive_dimensions,
    validate_rmsnorm_rope_inputs,
)


@triton.jit
def fused_rmsnorm_rope_kernel(
    x_ptr,
    output_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    stride_x_batch,
    stride_x_seq,
    stride_x_hidden,
    stride_out_batch,
    stride_out_seq,
    stride_out_hidden,
    stride_cos_seq,
    stride_cos_dim,
    batch_size,
    seq_len,
    hidden_dim,
    head_dim,
    num_heads,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RMSNorm + RoPE kernel.

    Combines RMSNorm and RoPE into a single kernel to minimize HBM access.
    Each program instance processes one row (one position in one batch).

    Memory access pattern:
    - Without fusion: HBM -> RMSNorm -> HBM -> RoPE -> HBM (3 HBM accesses)
    - With fusion: HBM -> [RMSNorm + RoPE in registers] -> HBM (1 HBM access)
    """
    # Get row index (batch * seq_len + seq_idx)
    row_idx = tl.program_id(0)
    batch_idx = row_idx // seq_len
    seq_idx = row_idx % seq_len

    # Skip if out of bounds
    if batch_idx >= batch_size:
        return

    # Compute base pointers
    x_row = x_ptr + batch_idx * stride_x_batch + seq_idx * stride_x_seq
    out_row = output_ptr + batch_idx * stride_out_batch + seq_idx * stride_out_seq
    cos_row = cos_ptr + seq_idx * stride_cos_seq
    sin_row = sin_ptr + seq_idx * stride_cos_seq

    # Step 1: Compute RMS (sum of squares)
    sum_sq = tl.zeros([1], dtype=tl.float32)

    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_dim
        x = tl.load(x_row + cols * stride_x_hidden, mask=mask, other=0.0)
        sum_sq += tl.sum(tl.where(mask, x.to(tl.float32) * x.to(tl.float32), 0.0))

    # Compute inverse RMS
    mean_sq = sum_sq / hidden_dim
    rrms = tl.rsqrt(mean_sq + eps)

    # Step 2: Apply RMSNorm and RoPE together
    half_head = head_dim // 2

    for head_idx in range(num_heads):
        head_offset = head_idx * head_dim

        # Process pairs for RoPE rotation
        for i in range(0, half_head, BLOCK_SIZE):
            cols = i + tl.arange(0, BLOCK_SIZE)
            mask = cols < half_head

            # Indices for first and second half of head
            idx1 = head_offset + cols
            idx2 = head_offset + half_head + cols

            # Load x values
            x1 = tl.load(x_row + idx1 * stride_x_hidden, mask=mask, other=0.0)
            x2 = tl.load(x_row + idx2 * stride_x_hidden, mask=mask, other=0.0)

            # Load weights
            w1 = tl.load(weight_ptr + idx1, mask=mask, other=0.0)
            w2 = tl.load(weight_ptr + idx2, mask=mask, other=0.0)

            # Apply RMSNorm
            x1_norm = x1.to(tl.float32) * rrms * w1.to(tl.float32)
            x2_norm = x2.to(tl.float32) * rrms * w2.to(tl.float32)

            # Load cos and sin
            cos_val = tl.load(cos_row + cols * stride_cos_dim, mask=mask, other=1.0)
            sin_val = tl.load(sin_row + cols * stride_cos_dim, mask=mask, other=0.0)

            # Apply RoPE rotation
            out1 = x1_norm * cos_val.to(tl.float32) - x2_norm * sin_val.to(tl.float32)
            out2 = x1_norm * sin_val.to(tl.float32) + x2_norm * cos_val.to(tl.float32)

            # Store results
            tl.store(out_row + idx1 * stride_out_hidden, out1.to(x1.dtype), mask=mask)
            tl.store(out_row + idx2 * stride_out_hidden, out2.to(x1.dtype), mask=mask)


def _largest_pow2_leq(n: int) -> int:
    """不超过 n 的最大 2 的幂（n >= 1 时结果 >= 1）。"""
    return 1 << (max(int(n), 1).bit_length() - 1)


def _require_pow2(value: int, name: str) -> None:
    """BLOCK_SIZE 必须是 2 的幂，否则 tl.arange 会在编译期崩溃。"""
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or (value & (value - 1)) != 0
    ):
        raise ValueError(f"{name} must be a power of two >= 1, got {value}")


def fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
    num_heads: int = None,
    BLOCK_SIZE: int = None,
    num_warps: int = None,
    num_stages: int = None,
) -> torch.Tensor:
    """Apply fused RMSNorm + RoPE transformation.

    This function combines RMSNorm and Rotary Position Embedding into a single
    kernel launch, reducing memory bandwidth requirements by eliminating
    intermediate HBM writes.

    Mathematical operations:
        1. RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
        2. RoPE: y_rope = y * cos + rotate_half(y) * sin

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim]
        weight: RMSNorm weight of shape [hidden_dim]
        cos: Cosine position embeddings of shape [seq_len, head_dim]
        sin: Sine position embeddings of shape [seq_len, head_dim]
        eps: Small constant for numerical stability (default: 1e-6)
        num_heads: Number of attention heads (inferred from hidden_dim/head_dim if not provided)

    Returns:
        Output tensor of shape [batch, seq_len, hidden_dim] with RMSNorm + RoPE applied

    Raises:
        DeviceError: If CUDA is not available
        ShapeMismatchError: If tensor shapes are incompatible
        UnsupportedDtypeError: If tensor dtypes are unsupported

    Example:
        >>> x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
        >>> weight = torch.ones(4096, device='cuda', dtype=torch.float16)
        >>> cos = torch.randn(128, 64, device='cuda', dtype=torch.float16)
        >>> sin = torch.randn(128, 64, device='cuda', dtype=torch.float16)
        >>> output = fused_rmsnorm_rope(x, weight, cos, sin)

    Note:
        All tensors must be on CUDA device and contiguous.
    """
    # Validate inputs
    batch_size, seq_len, hidden_dim, head_dim, num_heads = validate_rmsnorm_rope_inputs(
        x, weight, cos, sin, num_heads
    )

    # Additional validation
    validate_eps(eps)
    validate_head_dim(head_dim)
    validate_positive_dimensions(
        batch_size=batch_size, seq_len=seq_len, hidden_dim=hidden_dim, head_dim=head_dim
    )

    # Handle empty tensors
    if batch_size == 0 or seq_len == 0 or hidden_dim == 0:
        return torch.empty_like(x)

    # Handle 4D cos/sin format
    if cos.dim() == 4:
        cos = cos.squeeze(0).squeeze(1)  # [seq_len, head_dim]
        sin = sin.squeeze(0).squeeze(1)

    # Allocate output
    output = torch.empty_like(x)

    # BLOCK_SIZE 必须是 2 的幂（tl.arange 要求）。head_dim//2 非 2 的幂时
    # （如 head_dim=96 → 48）取不超过它的最大 2 的幂，越界由 kernel 的
    # cols < half_head 掩码兜底。
    if BLOCK_SIZE is None:
        BLOCK_SIZE = min(128, _largest_pow2_leq(head_dim // 2))
    _require_pow2(BLOCK_SIZE, "BLOCK_SIZE")

    # Launch kernel
    grid = (batch_size * seq_len,)
    launch_kwargs = {}
    if num_warps is not None:
        launch_kwargs["num_warps"] = num_warps
    if num_stages is not None:
        launch_kwargs["num_stages"] = num_stages

    fused_rmsnorm_rope_kernel[grid](
        x,
        output,
        weight,
        cos,
        sin,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        cos.stride(0),
        cos.stride(1),
        batch_size,
        seq_len,
        hidden_dim,
        head_dim,
        num_heads,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        **launch_kwargs,
    )

    return output


class FusedRMSNormRoPE(torch.nn.Module):
    """PyTorch module wrapper for fused RMSNorm + RoPE.

    This module provides a convenient interface for using the fused kernel
    in PyTorch models.

    Args:
        hidden_dim: Hidden dimension size
        head_dim: Head dimension for RoPE
        eps: Small constant for numerical stability
    """

    def __init__(self, hidden_dim: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.eps = eps
        self.num_heads = hidden_dim // head_dim

        # RMSNorm weight parameter
        self.weight = torch.nn.Parameter(torch.ones(hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply fused RMSNorm + RoPE.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            cos: Cosine position embeddings
            sin: Sine position embeddings

        Returns:
            Output tensor with RMSNorm + RoPE applied
        """
        return fused_rmsnorm_rope(x, self.weight, cos, sin, self.eps, self.num_heads)
