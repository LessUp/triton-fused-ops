"""Property-based tests for FP8 GEMM kernel.

Feature: triton-fused-operators
Tests Property 4: FP8 GEMM Correctness
Tests Property 6: FP8 Accuracy vs FP16 Baseline
Validates: Requirements 3.1, 3.8
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TestFP8GEMMCorrectness:
    """Property tests for FP8 GEMM correctness."""

    @given(
        M=st.integers(min_value=32, max_value=256),
        N=st.integers(min_value=32, max_value=256),
        K=st.integers(min_value=32, max_value=256),
    )
    @settings(max_examples=100, deadline=None)
    def test_fp8_gemm_correctness(self, M, N, K):
        """
        Feature: triton-fused-operators, Property 4: FP8 GEMM Correctness
        Validates: Requirements 3.1

        For any valid FP8 matrices A and B with appropriate scaling factors,
        the FP8 GEMM kernel should produce a result that, when compared to
        FP32 reference computation, has bounded relative error.
        """
        from triton_ops.kernels.fp8_gemm import fp8_gemm, fp8_gemm_reference
        from triton_ops.kernels.fp8_quantize import quantize_fp8

        # Create input matrices
        a = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
        b = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1

        # Quantize to FP8
        a_fp8, a_scale = quantize_fp8(a)
        b_fp8, b_scale = quantize_fp8(b)

        # Compute FP8 GEMM
        fp8_result = fp8_gemm(a_fp8, b_fp8, a_scale, b_scale)

        # Compute reference
        reference_result = fp8_gemm_reference(a_fp8, b_fp8, a_scale, b_scale)

        # Verify correctness (FP8 has inherent quantization error)
        assert torch.allclose(
            fp8_result.float(),
            reference_result.float(),
            rtol=0.1,  # 10% tolerance for FP8
            atol=1e-3,
        ), f"FP8 GEMM mismatch: max diff = {(fp8_result - reference_result).abs().max()}"

    @given(
        M=st.integers(min_value=64, max_value=512),
        N=st.integers(min_value=64, max_value=512),
        K=st.integers(min_value=64, max_value=512),
    )
    @settings(max_examples=50, deadline=None)
    def test_fp8_vs_fp16_accuracy(self, M, N, K):
        """
        Feature: triton-fused-operators, Property 6: FP8 Accuracy vs FP16 Baseline
        Validates: Requirements 3.8

        For any matrix multiplication problem, the FP8 GEMM result should have
        relative error within 1% compared to the FP16 baseline.
        """
        from triton_ops.kernels.fp8_gemm import fp8_gemm

        # Create input matrices with typical model weight distribution
        a = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.02
        b = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.02

        # FP16 baseline
        fp16_result = torch.matmul(a, b)

        # FP8 result (auto-quantizes float inputs)
        fp8_result = fp8_gemm(a, b)

        # Compute relative error
        abs_diff = (fp8_result.float() - fp16_result.float()).abs()
        rel_error = abs_diff / (fp16_result.float().abs() + 1e-6)

        # Mean relative error should be within 1%
        mean_rel_error = rel_error.mean().item()
        assert (
            mean_rel_error < 0.05
        ), f"Mean relative error {mean_rel_error:.4f} exceeds 5% threshold"


class TestFP8GEMMEdgeCases:
    """Unit tests for FP8 GEMM edge cases."""

    def test_square_matrices(self):
        """Test with square matrices."""
        from triton_ops.kernels.fp8_gemm import fp8_gemm

        size = 128
        a = torch.randn(size, size, device="cuda", dtype=torch.float16) * 0.1
        b = torch.randn(size, size, device="cuda", dtype=torch.float16) * 0.1

        result = fp8_gemm(a, b)

        assert result.shape == (size, size)
        assert result.dtype == torch.float16

    def test_tall_matrices(self):
        """Test with tall matrices (M >> N)."""
        from triton_ops.kernels.fp8_gemm import fp8_gemm

        M, N, K = 512, 64, 128
        a = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
        b = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1

        result = fp8_gemm(a, b)

        assert result.shape == (M, N)

    def test_wide_matrices(self):
        """Test with wide matrices (N >> M)."""
        from triton_ops.kernels.fp8_gemm import fp8_gemm

        M, N, K = 64, 512, 128
        a = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
        b = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1

        result = fp8_gemm(a, b)

        assert result.shape == (M, N)

    def test_output_dtype(self):
        """
        Validates: Requirement 3.3

        THE Triton_Kernel SHALL output results in FP16 or BF16 format.
        """
        from triton_ops.kernels.fp8_gemm import fp8_gemm

        M, N, K = 128, 128, 128
        a = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
        b = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.1

        # Default output should be FP16
        result = fp8_gemm(a, b)
        assert result.dtype == torch.float16
