"""Triton SGEMM kernel implementation.

This module implements a classic blocked SGEMM kernel written with Triton:
C[M, N] = A[M, K] @ B[K, N]

The kernel follows the canonical Triton tutorial structure:
- each program computes one BLOCK_M x BLOCK_N tile of the output,
- the K dimension is accumulated in BLOCK_K-sized chunks,
- all loads/stores are masked so that M/N/K need not be multiples of the
  block sizes (supports non-power-of-two, non-tile-aligned shapes).

Supported dtypes: float32, float16, bfloat16 (accumulation in float32).
"""

import torch
import triton
import triton.language as tl

from trifuse.exceptions import (
    DeviceError,
    ShapeMismatchError,
    UnsupportedDtypeError,
)

# Supported dtypes for SGEMM
SUPPORTED_SGEMM_DTYPES = [torch.float16, torch.bfloat16, torch.float32]


@triton.jit
def sgemm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Blocked SGEMM kernel.

    Each program computes one BLOCK_M x BLOCK_N output tile:
        C[offs_m, offs_n] += A[offs_m, :] @ B[:, offs_n]

    Args:
        A: Row-major [M, K] input.
        B: Row-major [K, N] input.
        C: Row-major [M, N] output (accumulated in fp32, stored in C dtype).
        M/N/K: Matrix dimensions.
        stride_*: Row/column strides for A, B and C.
        BLOCK_M/BLOCK_N/BLOCK_K: Tile sizes (constexpr).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # fp32 accumulator (all dtypes accumulate in fp32)
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # input_precision="ieee": 对 fp32 输入禁用 TF32 截断，保持 SGEMM 的
        # fp32 精度（TF32 只影响 fp32；fp16/bf16 路径不受该参数影响）。
        acc = tl.dot(a, b, acc, input_precision="ieee")

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)


def _validate_sgemm_inputs(a: torch.Tensor, b: torch.Tensor):
    """Validate SGEMM inputs, returning (M, N, K).

    Raises:
        DeviceError: If any tensor is not on CUDA.
        UnsupportedDtypeError: If any tensor dtype is unsupported or mismatched.
        ShapeMismatchError: If tensors are not 2-D row-major or inner dims mismatch.
        ValueError: If a tensor is not contiguous.
    """
    for name, tensor in (("a", a), ("b", b)):
        if not tensor.is_cuda:
            raise DeviceError(
                f"{name} must be on CUDA device, got {tensor.device}",
                expected_device="cuda",
                actual_device=str(tensor.device),
                tensor_name=name,
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.dtype not in SUPPORTED_SGEMM_DTYPES:
            raise UnsupportedDtypeError(
                f"{name} has unsupported dtype {tensor.dtype}, supported: {SUPPORTED_SGEMM_DTYPES}",
                dtype=tensor.dtype,
                supported_dtypes=SUPPORTED_SGEMM_DTYPES,
                tensor_name=name,
            )
        if tensor.ndim != 2:
            raise ShapeMismatchError(
                f"{name} must be a 2-D row-major matrix, got {tensor.ndim} dims",
                expected=(2,),
                actual=tuple(tensor.shape),
                tensor_name=name,
            )

    if a.dtype != b.dtype:
        raise UnsupportedDtypeError(
            f"a and b dtypes must match; got {a.dtype} and {b.dtype}",
            dtype=a.dtype,
            supported_dtypes=SUPPORTED_SGEMM_DTYPES,
            tensor_name="a/b",
        )
    if a.shape[1] != b.shape[0]:
        raise ShapeMismatchError(
            f"inner dimensions mismatch: a is {tuple(a.shape)}, b is {tuple(b.shape)}",
            expected=(a.shape[0], a.shape[1], b.shape[1]),
            actual=(a.shape[0], a.shape[1], b.shape[1]),
            tensor_name="a/b",
        )

    M, K = a.shape
    N = b.shape[1]
    if M < 1 or N < 1 or K < 1:
        raise ShapeMismatchError(
            f"M, N, K must be >= 1, got M={M}, N={N}, K={K}",
            expected=(M, N, K),
            actual=(M, N, K),
            tensor_name="a/b",
        )
    return M, N, K


def sgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """Blocked SGEMM: C = A @ B, computed with a Triton kernel.

    Args:
        a: Row-major CUDA tensor of shape [M, K].
        b: Row-major CUDA tensor of shape [K, N].
        BLOCK_M/BLOCK_N/BLOCK_K: Tile sizes (must be powers of two).

    Returns:
        Row-major CUDA tensor of shape [M, N] with the same dtype as `a`.

    Raises:
        DeviceError: If any tensor is not on CUDA.
        UnsupportedDtypeError: If any tensor dtype is unsupported or mismatched.
        ShapeMismatchError: If tensors are not 2-D or inner dims mismatch.
        ValueError: If a tensor is not contiguous.

    Example:
        >>> a = torch.randn(128, 256, device='cuda', dtype=torch.float16)
        >>> b = torch.randn(256, 64, device='cuda', dtype=torch.float16)
        >>> c = sgemm(a, b)
    """
    M, N, K = _validate_sgemm_inputs(a, b)

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    sgemm_kernel[grid](
        a,
        b,
        c,
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
        num_warps=4,
        num_stages=3,
    )

    return c
