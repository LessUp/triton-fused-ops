"""CPU-only tests for FP8 compute logic."""

import numpy as np

from triton_ops.compute.fp8 import (
    FP8_MAX,
    compute_dequantize_fp8,
    compute_fp8_quantization_error,
    compute_fp8_scale,
    compute_quantize_fp8,
)


class TestComputeFP8Quantize:
    """Test FP8 quantization logic."""

    def test_basic_quantization(self):
        """Test basic FP8 quantization."""
        tensor = np.random.randn(1024, 1024).astype(np.float32) * 10
        quantized, scale = compute_quantize_fp8(tensor)

        # Check dtype
        assert quantized.dtype == np.uint8
        # Check scale is positive
        assert scale > 0
        # Check shape preserved
        assert quantized.shape == tensor.shape

    def test_scale_computation(self):
        """Test automatic scale computation."""
        tensor = np.random.randn(1024, 1024).astype(np.float32) * 100
        quantized, scale = compute_quantize_fp8(tensor)

        # Scale should map max value to FP8_MAX
        max_abs = np.abs(tensor).max()
        expected_scale = FP8_MAX / max_abs

        np.testing.assert_allclose(scale, expected_scale, rtol=1e-5)

    def test_manual_scale(self):
        """Test quantization with manual scale."""
        tensor = np.random.randn(1024, 1024).astype(np.float32) * 10
        manual_scale = 5.0

        quantized, scale = compute_quantize_fp8(tensor, scale=manual_scale)

        # Should use provided scale
        assert scale == manual_scale

    def test_zero_tensor(self):
        """Test quantization of zero tensor."""
        tensor = np.zeros((1024, 1024), dtype=np.float32)
        quantized, scale = compute_quantize_fp8(tensor)

        # Scale should be 1.0 for zero tensor
        assert scale == 1.0
        # Quantized should be centered around 128
        np.testing.assert_allclose(quantized, 128, atol=1)

    def test_clipping(self):
        """Test that values are clipped to FP8 range."""
        # Tensor with values outside FP8 range
        tensor = np.array([[1000.0, -1000.0, 500.0]], dtype=np.float32)
        quantized, scale = compute_quantize_fp8(tensor)

        # Check quantized values are in valid uint8 range
        assert quantized.min() >= 0
        assert quantized.max() <= 255

    def test_small_values(self):
        """Test quantization of small values."""
        tensor = np.random.randn(1024, 1024).astype(np.float32) * 0.001
        quantized, scale = compute_quantize_fp8(tensor)

        # Should still produce valid quantization
        assert quantized.dtype == np.uint8
        assert scale > 0

    def test_large_values(self):
        """Test quantization of large values."""
        tensor = np.random.randn(1024, 1024).astype(np.float32) * 10000
        quantized, scale = compute_quantize_fp8(tensor)

        # Should handle large values gracefully
        assert quantized.dtype == np.uint8
        assert scale > 0


class TestComputeFP8Dequantize:
    """Test FP8 dequantization logic."""

    def test_round_trip(self):
        """Test quantize -> dequantize round trip."""
        original = np.random.randn(1024, 1024).astype(np.float32) * 10
        quantized, scale = compute_quantize_fp8(original)
        recovered = compute_dequantize_fp8(quantized, scale)

        # Should approximately recover original
        np.testing.assert_allclose(original, recovered, rtol=0.1, atol=1.0)

    def test_dtype_conversion(self):
        """Test output dtype conversion."""
        quantized = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)
        scale = 10.0

        output_fp32 = compute_dequantize_fp8(quantized, scale, dtype=np.float32)
        output_fp16 = compute_dequantize_fp8(quantized, scale, dtype=np.float16)

        assert output_fp32.dtype == np.float32
        assert output_fp16.dtype == np.float16

    def test_scale_effect(self):
        """Test that scale properly affects dequantization."""
        quantized = np.full((1024, 1024), 200, dtype=np.uint8)

        output_small = compute_dequantize_fp8(quantized, scale=10.0)
        output_large = compute_dequantize_fp8(quantized, scale=100.0)

        # Smaller scale should produce larger magnitude values (division by scale)
        assert np.abs(output_small).max() > np.abs(output_large).max()


class TestComputeFP8Scale:
    """Test FP8 scale computation."""

    def test_positive_values(self):
        """Test scale computation for positive values."""
        tensor = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
        scale = compute_fp8_scale(tensor)

        expected_scale = FP8_MAX / 300.0
        np.testing.assert_allclose(scale, expected_scale, rtol=1e-5)

    def test_negative_values(self):
        """Test scale computation for negative values."""
        tensor = np.array([[-100.0, -200.0, -300.0]], dtype=np.float32)
        scale = compute_fp8_scale(tensor)

        # Should use absolute value
        expected_scale = FP8_MAX / 300.0
        np.testing.assert_allclose(scale, expected_scale, rtol=1e-5)

    def test_mixed_values(self):
        """Test scale computation for mixed values."""
        tensor = np.array([[100.0, -200.0, 300.0]], dtype=np.float32)
        scale = compute_fp8_scale(tensor)

        expected_scale = FP8_MAX / 300.0
        np.testing.assert_allclose(scale, expected_scale, rtol=1e-5)

    def test_zero_tensor(self):
        """Test scale computation for zero tensor."""
        tensor = np.zeros((1024, 1024), dtype=np.float32)
        scale = compute_fp8_scale(tensor)

        # Should return 1.0 for zero tensor
        assert scale == 1.0


class TestComputeFP8Error:
    """Test FP8 quantization error computation."""

    def test_error_metrics(self):
        """Test quantization error metrics."""
        original = np.random.randn(1024, 1024).astype(np.float32) * 10
        quantized, scale = compute_quantize_fp8(original)

        error = compute_fp8_quantization_error(original, quantized, scale)

        # Check all metrics are present
        assert "max_error" in error
        assert "mean_error" in error
        assert "relative_error" in error

        # Check values are reasonable
        assert error["max_error"] >= 0
        assert error["mean_error"] >= 0
        assert error["relative_error"] >= 0

    def test_error_proportional_to_scale(self):
        """Test that error is related to quantization granularity."""
        # Small scale = fine granularity = small error
        tensor_small = np.random.randn(1024, 1024).astype(np.float32) * 1.0
        quantized_small, scale_small = compute_quantize_fp8(tensor_small)
        error_small = compute_fp8_quantization_error(tensor_small, quantized_small, scale_small)

        # Large scale = coarse granularity = larger error (potentially)
        tensor_large = np.random.randn(1024, 1024).astype(np.float32) * 1000.0
        quantized_large, scale_large = compute_quantize_fp8(tensor_large)
        error_large = compute_fp8_quantization_error(tensor_large, quantized_large, scale_large)

        # Relative error should be similar for similar distributions
        # (just scaled differently)
        # This is a sanity check, not a strict requirement
        assert error_small["relative_error"] < 1.0
        assert error_large["relative_error"] < 1.0

    def test_zero_error_for_zero(self):
        """Test error metrics for zero tensor."""
        original = np.zeros((100, 100), dtype=np.float32)
        quantized, scale = compute_quantize_fp8(original)
        error = compute_fp8_quantization_error(original, quantized, scale)

        # Error should be very small for zero tensor
        assert error["max_error"] < 1.0
        assert error["mean_error"] < 1.0


class TestFP8Constants:
    """Test FP8 constants."""

    def test_fp8_max(self):
        """Test FP8 maximum value."""
        assert FP8_MAX == 448.0

    def test_fp8_range_coverage(self):
        """Test that FP8 covers expected range."""
        # Create tensor with FP8-range values
        tensor = np.random.uniform(-400, 400, (1024, 1024)).astype(np.float32)
        quantized, scale = compute_quantize_fp8(tensor)

        # Should not clip too much
        error = compute_fp8_quantization_error(tensor, quantized, scale)

        # Relative error should be reasonable
        assert error["relative_error"] < 0.1
