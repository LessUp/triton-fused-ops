"""FP8 GEMM (General Matrix Multiplication) Triton kernel.

This module implements FP8 matrix multiplication with:
- FP8 E4M3 inputs
- FP32 accumulation for numerical stability
- FP16/BF16 output
- Block pointer optimization for efficient memory access
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from triton_ops.exceptions import DeviceError
from triton_ops.kernels.fp8_quantize import quantize_fp8
from triton_ops.validation import validate_fp8_gemm_inputs

# FP8 constants
FP8_MAX = 448.0


@triton.jit
def fp8_gemm_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    # Scale factors
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # Strides
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
):
    """FP8 Matrix Multiplication kernel.

    Computes C = A @ B where A and B are in FP8 format.

    - Inputs: FP8 E4M3 (stored as uint8)
    - Accumulation: FP32 for numerical stability
    - Output: FP16 or BF16

    Uses grouped ordering for better L2 cache utilization.
    """
    # Program ID
    pid = tl.program_id(0)

    # Number of tiles in each dimension
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    # Compute tile indices with grouped ordering
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Compute starting row and column
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Load scale factors
    a_scale = tl.load(a_scale_ptr)
    b_scale = tl.load(b_scale_ptr)
    # Avoid division by zero
    scale_product = a_scale * b_scale
    inv_scale = tl.where(scale_product != 0, 1.0 / scale_product, 0.0)

    # Pointers to first block of A and B
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Main loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load A block (FP8 as uint8)
        a_mask = (rm[:, None] < M) & ((k + rk[None, :]) < K)
        a_uint8 = tl.load(a_ptrs, mask=a_mask, other=128)

        # Load B block (FP8 as uint8)
        b_mask = ((k + rk[:, None]) < K) & (rn[None, :] < N)
        b_uint8 = tl.load(b_ptrs, mask=b_mask, other=128)

        # Convert FP8 to float
        # FP8 is stored as uint8 with 128 offset (signed range shifted to unsigned)
        a_int = a_uint8.to(tl.int32) - 128
        b_int = b_uint8.to(tl.int32) - 128

        a_float = a_int.to(tl.float32) / 127.0 * FP8_MAX
        b_float = b_int.to(tl.float32) / 127.0 * FP8_MAX

        # Accumulate matrix product
        acc += tl.dot(a_float, b_float)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Apply inverse scale and convert to output dtype
    acc = acc * inv_scale

    # Store output with correct dtype
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    if OUTPUT_BF16:
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)
    else:
        tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Perform FP8 matrix multiplication.

    If inputs are not already in FP8 format, they will be quantized automatically.
    Uses FP32 accumulation for numerical stability.

    Args:
        a: First matrix of shape [M, K] - can be FP8 (uint8) or float
        b: Second matrix of shape [K, N] - can be FP8 (uint8) or float
        a_scale: Scale factor for A (required if A is FP8, computed if float)
        b_scale: Scale factor for B (required if B is FP8, computed if float)
        output_dtype: Output data type - float16 (default) or bfloat16

    Returns:
        Result matrix of shape [M, N] in output_dtype

    Raises:
        DeviceError: If CUDA is not available
        ShapeMismatchError: If tensor shapes are incompatible
        UnsupportedDtypeError: If tensor dtypes are unsupported

    Example:
        >>> a = torch.randn(512, 1024, device='cuda', dtype=torch.float16)
        >>> b = torch.randn(1024, 2048, device='cuda', dtype=torch.float16)
        >>> c = fp8_gemm(a, b)  # Auto-quantizes to FP8

    Note:
        All tensors must be on CUDA device and contiguous.
    """
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise DeviceError(
            "CUDA is not available. This kernel requires a CUDA-capable GPU.",
            expected_device="cuda",
            actual_device="cpu",
        )
    # Handle float inputs by quantizing to FP8
    if a.dtype in [torch.float16, torch.bfloat16, torch.float32]:
        a, a_scale = quantize_fp8(a)
    if b.dtype in [torch.float16, torch.bfloat16, torch.float32]:
        b, b_scale = quantize_fp8(b)

    # Validate inputs
    M, N, K = validate_fp8_gemm_inputs(a, b, a_scale, b_scale, output_dtype)

    # Handle empty tensors
    if M == 0 or N == 0 or K == 0:
        return torch.empty(M, N, dtype=output_dtype, device=a.device)

    # Allocate output
    c = torch.empty(M, N, dtype=output_dtype, device=a.device)

    # Select block sizes - use heuristics based on problem size for better performance
    # Larger blocks for larger matrices
    if M >= 2048 and N >= 2048:
        BLOCK_M, BLOCK_N = 128, 128
    elif M >= 1024 or N >= 1024:
        BLOCK_M, BLOCK_N = 64, 128
    else:
        BLOCK_M, BLOCK_N = 64, 64

    # K dimension block size
    BLOCK_K = 64 if K >= 1024 else 32

    # Group size for better L2 cache utilization
    GROUP_SIZE_M = 8

    # Determine output type flag
    output_bf16 = output_dtype == torch.bfloat16

    # Grid size
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    # Launch kernel
    fp8_gemm_kernel[grid](
        a,
        b,
        c,
        a_scale,
        b_scale,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        OUTPUT_BF16=output_bf16,
    )

    return c


def fp8_gemm_reference(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor = None,
    b_scale: torch.Tensor = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reference implementation of FP8 GEMM for testing.

    Args:
        a: First matrix (FP8 as uint8 or float)
        b: Second matrix (FP8 as uint8 or float)
        a_scale: Scale factor for A
        b_scale: Scale factor for B
        output_dtype: Output dtype

    Returns:
        Result matrix
    """
    # Handle FP8 inputs
    if a.dtype == torch.uint8:
        a_int = a.to(torch.int32) - 128
        a_float = a_int.float() / 127.0 * FP8_MAX / a_scale
    else:
        a_float = a.float()

    if b.dtype == torch.uint8:
        b_int = b.to(torch.int32) - 128
        b_float = b_int.float() / 127.0 * FP8_MAX / b_scale
    else:
        b_float = b.float()

    # Compute matrix multiplication in FP32
    c = torch.matmul(a_float, b_float)

    return c.to(output_dtype)


class FP8Linear(torch.nn.Module):
    """Linear layer with FP8 quantized weights.

    This module stores weights in FP8 format and performs FP8 GEMM
    during forward pass.

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        bias: Whether to include bias (default: False)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight in FP16, will be quantized on first forward
        self.register_buffer("weight_fp8", None)
        self.register_buffer("weight_scale", None)

        # Original weight for initialization
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features) * 0.02)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self._quantized = False

    def quantize_weights(self):
        """Quantize weights to FP8 format and cache the transposed version."""
        if not self._quantized:
            weight_fp8, weight_scale = quantize_fp8(self.weight.data)
            self.weight_fp8 = weight_fp8
            self.weight_scale = weight_scale
            # Pre-transpose and cache — avoids .t().contiguous() on every forward
            self.register_buffer("weight_fp8_t", weight_fp8.t().contiguous())
            self._quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with FP8 GEMM.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Quantize weights if not done
        self.quantize_weights()

        # Reshape input for GEMM
        orig_shape = x.shape
        x_2d = x.view(-1, self.in_features)

        # Quantize input
        x_fp8, x_scale = quantize_fp8(x_2d)

        # FP8 GEMM: x @ weight.T (pre-transposed and cached)
        output = fp8_gemm(
            x_fp8,
            self.weight_fp8_t,
            x_scale,
            self.weight_scale,
            output_dtype=x.dtype,
        )

        # Reshape output
        output = output.view(*orig_shape[:-1], self.out_features)

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output
