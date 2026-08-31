"""Differential and boundary tests for the Triton SGEMM kernel.

Feature: triton-fused-operators
Validates: SGEMM correctness vs torch.mm, boundary shapes, and failure paths.
"""

import pytest
import torch

from trifuse.exceptions import DeviceError, ShapeMismatchError, UnsupportedDtypeError
from trifuse.kernels.sgemm import sgemm

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# (M, N, K) shapes: aligned square, M=1, N=1, and a fully non-aligned shape
DIFF_SHAPES = [(64, 64, 64), (1, 128, 896), (128, 1, 896), (17, 33, 65)]


@pytest.mark.parametrize("M,N,K", DIFF_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_sgemm_matches_torch_mm(M, N, K, dtype):
    """SGEMM output must match torch.mm within fp16 tolerances.

    rtol/atol=1e-2 covers fp16 accumulation error; fp32 is far below that.
    """
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)

    out = sgemm(a, b)
    ref = a @ b

    assert out.shape == (M, N)
    assert out.dtype == dtype
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2), (
        f"SGEMM mismatch for ({M},{N},{K}) {dtype}: "
        f"max abs diff = {(out.float() - ref.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("M,N,K", [(1, 64, 64), (64, 1, 64), (64, 64, 1), (17, 33, 65)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_sgemm_boundary_shapes(M, N, K, dtype):
    """Boundary shapes: M=1, N=1, K=1 and non-power-of-two dims."""
    torch.manual_seed(1)
    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)

    out = sgemm(a, b)
    ref = a @ b

    assert out.shape == (M, N)
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2), (
        f"SGEMM boundary mismatch for ({M},{N},{K}) {dtype}"
    )


def test_sgemm_bf16():
    """bfloat16 is a supported dtype and must match torch.mm."""
    torch.manual_seed(2)
    a = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 48, device="cuda", dtype=torch.bfloat16)

    out = sgemm(a, b)
    ref = a @ b
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_sgemm_custom_tile_sizes():
    """Custom tile sizes must still produce correct results."""
    torch.manual_seed(3)
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 128, device="cuda", dtype=torch.float16)

    out = sgemm(a, b, BLOCK_M=128, BLOCK_N=32, BLOCK_K=64)
    ref = a @ b
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_sgemm_cpu_tensor_raises():
    """CPU tensors must be rejected with DeviceError."""
    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    with pytest.raises(DeviceError):
        sgemm(a, b)


def test_sgemm_wrong_dtype_raises():
    """Unsupported dtype must be rejected with UnsupportedDtypeError."""
    a = torch.ones(4, 4, device="cuda", dtype=torch.int32)
    b = torch.ones(4, 4, device="cuda", dtype=torch.int32)
    with pytest.raises(UnsupportedDtypeError):
        sgemm(a, b)


def test_sgemm_mismatched_dtypes_raises():
    """Mismatched input dtypes must be rejected."""
    a = torch.randn(4, 4, device="cuda", dtype=torch.float16)
    b = torch.randn(4, 4, device="cuda", dtype=torch.float32)
    with pytest.raises(UnsupportedDtypeError):
        sgemm(a, b)


def test_sgemm_non_contiguous_raises():
    """Non-contiguous tensors must be rejected with ValueError."""
    a = torch.randn(8, 8, device="cuda", dtype=torch.float16)[:, ::2]
    b = torch.randn(4, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="contiguous"):
        sgemm(a, b)


def test_sgemm_inner_dim_mismatch_raises():
    """Inner dimension mismatch must be rejected with ShapeMismatchError."""
    a = torch.randn(8, 4, device="cuda", dtype=torch.float16)
    b = torch.randn(5, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(ShapeMismatchError):
        sgemm(a, b)


def test_sgemm_1d_tensor_raises():
    """Non-2-D tensors must be rejected with ShapeMismatchError."""
    a = torch.randn(8, device="cuda", dtype=torch.float16)
    b = torch.randn(8, device="cuda", dtype=torch.float16)
    with pytest.raises(ShapeMismatchError):
        sgemm(a, b)
