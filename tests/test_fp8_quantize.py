"""Property-based tests for FP8 quantization.

Feature: triton-fused-operators
Tests Property 5: FP8 Quantization Round-Trip
Validates: Requirements 3.4
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TestFP8QuantizationRoundTrip:
    """Property tests for FP8 quantization round-trip consistency."""

    @given(
        M=st.integers(min_value=16, max_value=512),
        N=st.integers(min_value=16, max_value=512),
    )
    @settings(max_examples=100, deadline=None)
    def test_fp8_round_trip(self, M, N):
        """
        Feature: triton-fused-operators, Property 5: FP8 Quantization Round-Trip
        Validates: Requirements 3.4

        For any valid FP16/BF16 tensor within the representable FP8 range,
        quantizing to FP8 and then dequantizing back should produce a result
        within the expected quantization error bounds.
        """
        from triton_ops.kernels.fp8_quantize import dequantize_fp8, quantize_fp8

        # Create input tensor with values in FP8 range
        # Scale to be within FP8 representable range
        tensor = torch.randn(M, N, device="cuda", dtype=torch.float16) * 10.0

        # Quantize
        quantized, scale = quantize_fp8(tensor)

        # Dequantize
        dequantized = dequantize_fp8(quantized, scale)

        # Verify round-trip error is bounded
        # FP8 E4M3 has ~3 bits of mantissa, so expect ~12.5% max error
        max_quantization_error = 0.15  # 15% relative error tolerance

        # Compute relative error (avoid division by zero)
        abs_diff = (tensor.float() - dequantized.float()).abs()
        rel_error = abs_diff / (tensor.float().abs() + 1e-6)

        # Most values should be within tolerance
        within_tolerance = (rel_error < max_quantization_error).float().mean()
        assert (
            within_tolerance > 0.95
        ), f"Only {within_tolerance*100:.1f}% of values within tolerance"

    @given(
        numel=st.integers(min_value=100, max_value=10000),
    )
    @settings(max_examples=50, deadline=None)
    def test_scale_computation(self, numel):
        """
        Test that scale computation correctly maps tensor range to FP8 range.
        """
        from triton_ops.models import FP8Format

        # Create tensor with known range
        tensor = torch.randn(numel, device="cuda", dtype=torch.float16) * 100.0

        # Compute scale
        scale = FP8Format.compute_scale(tensor)

        # Verify scaled tensor is within FP8 range
        scaled = tensor.float() * scale
        assert (
            scaled.abs().max() <= FP8Format.max_value * 1.01
        ), "Scaled tensor exceeds FP8 max value"


class TestFP8OverflowHandling:
    """Unit tests for FP8 overflow handling."""

    def test_overflow_detection(self):
        """
        Validates: Requirement 3.5

        IF overflow is detected during FP8 conversion, THEN the kernel SHALL
        dynamically adjust the scaling factor.
        """
        from triton_ops.kernels.fp8_quantize import quantize_fp8_with_overflow_handling

        # Create tensor with values that would overflow with scale=1
        tensor = torch.full((100,), 1000.0, device="cuda", dtype=torch.float16)

        # Should handle overflow by adjusting scale
        quantized, scale = quantize_fp8_with_overflow_handling(tensor)

        # Verify scale was adjusted
        assert scale < 1.0, "Scale should be reduced to handle overflow"

        # Verify no actual overflow in output
        assert not torch.isinf(quantized.float()).any(), "Output should not contain Inf"

    def test_zero_tensor(self):
        """Test quantization of zero tensor."""
        from triton_ops.kernels.fp8_quantize import dequantize_fp8, quantize_fp8

        tensor = torch.zeros(100, device="cuda", dtype=torch.float16)

        quantized, scale = quantize_fp8(tensor)
        dequantized = dequantize_fp8(quantized, scale)

        # Should remain zero
        assert torch.allclose(dequantized, tensor, atol=1e-6)

    def test_small_values(self):
        """Test quantization of very small values."""
        from triton_ops.kernels.fp8_quantize import dequantize_fp8, quantize_fp8

        tensor = torch.randn(100, device="cuda", dtype=torch.float16) * 1e-4

        quantized, scale = quantize_fp8(tensor)
        dequantized = dequantize_fp8(quantized, scale)

        # Should preserve sign at minimum
        signs_match = (tensor.sign() == dequantized.sign()) | (tensor == 0) | (dequantized == 0)
        assert signs_match.float().mean() > 0.9, "Signs should mostly be preserved"
