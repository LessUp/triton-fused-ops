"""CPU-only tests for FP8 reference implementation."""

import numpy as np

from triton_ops.models import FP8Format
from triton_ops.reference import dequantize_fp8, quantize_fp8
from triton_ops.reference.fp8 import compute_fp8_scale


class TestReferenceFP8Quantize:
    """Test FP8 quantization reference implementation."""

    def test_basic_quantization_cpu(self):
        """Test basic FP8 quantization on CPU."""
        tensor = np.random.randn(1024, 1024).astype(np.float32) * 10
        quantized, scale = quantize_fp8(tensor, backend="cpu")

        # Check dtype
        assert quantized.dtype == np.uint8
        # Check scale is positive
        assert scale > 0
        # Check shape preserved
        assert quantized.shape == tensor.shape

    def test_scale_computation_cpu(self):
        """Test automatic scale computation on CPU."""
        tensor = np.random.randn(1024, 1024).astype(np.float32) * 100
        quantized, scale = quantize_fp8(tensor, backend="cpu")

        # Scale should map max value to FP8_MAX
        max_abs = np.abs(tensor).max()
        expected_scale = FP8Format.max_value / max_abs

        np.testing.assert_allclose(scale, expected_scale, rtol=1e-5)

    def test_manual_scale_cpu(self):
        """Test quantization with manual scale on CPU."""
        tensor = np.random.randn(1024, 1024).astype(np.float32) * 10
        manual_scale = 5.0

        quantized, scale = quantize_fp8(tensor, scale=manual_scale, backend="cpu")

        # Should use provided scale
        assert scale == manual_scale

    def test_zero_tensor_cpu(self):
        """Test quantization of zero tensor on CPU."""
        tensor = np.zeros((1024, 1024), dtype=np.float32)
        quantized, scale = quantize_fp8(tensor, backend="cpu")

        # Scale should be 1.0 for zero tensor
        assert scale == 1.0
        # Quantized should be centered around 128
        np.testing.assert_allclose(quantized, 128, atol=1)

    def test_clipping_cpu(self):
        """Test that values are clipped to FP8 range on CPU."""
        # Tensor with values outside FP8 range
        tensor = np.array([[1000.0, -1000.0, 500.0]], dtype=np.float32)
        quantized, scale = quantize_fp8(tensor, backend="cpu")

        # Check quantized values are in valid uint8 range
        assert quantized.min() >= 0
        assert quantized.max() <= 255

    def test_small_values_cpu(self):
        """Test quantization of small values on CPU."""
        tensor = np.random.randn(1024, 1024).astype(np.float32) * 0.001
        quantized, scale = quantize_fp8(tensor, backend="cpu")

        # Should still produce valid quantization
        assert quantized.dtype == np.uint8
        assert scale > 0

    def test_large_values_cpu(self):
        """Test quantization of large values on CPU."""
        tensor = np.random.randn(1024, 1024).astype(np.float32) * 10000
        quantized, scale = quantize_fp8(tensor, backend="cpu")

        # Should handle large values gracefully
        assert quantized.dtype == np.uint8
        assert scale > 0


class TestReferenceFP8Dequantize:
    """Test FP8 dequantization reference implementation."""

    def test_round_trip_cpu(self):
        """Test quantize -> dequantize round trip on CPU."""
        original = np.random.randn(1024, 1024).astype(np.float32) * 10
        quantized, scale = quantize_fp8(original, backend="cpu")
        recovered = dequantize_fp8(quantized, scale, backend="cpu")

        # Should approximately recover original
        np.testing.assert_allclose(original, recovered, rtol=0.1, atol=1.0)

    def test_dtype_conversion_cpu(self):
        """Test output dtype conversion on CPU."""
        quantized = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)
        scale = 10.0

        output_fp32 = dequantize_fp8(quantized, scale, output_dtype=np.float32, backend="cpu")
        output_fp16 = dequantize_fp8(quantized, scale, output_dtype=np.float16, backend="cpu")

        assert output_fp32.dtype == np.float32
        assert output_fp16.dtype == np.float16

    def test_scale_effect_cpu(self):
        """Test that scale properly affects dequantization on CPU."""
        quantized = np.full((1024, 1024), 200, dtype=np.uint8)

        output_small = dequantize_fp8(quantized, scale=10.0, backend="cpu")
        output_large = dequantize_fp8(quantized, scale=100.0, backend="cpu")

        # Smaller scale should produce larger magnitude values (division by scale)
        assert np.abs(output_small).max() > np.abs(output_large).max()


class TestReferenceFP8Scale:
    """Test FP8 scale computation."""

    def test_positive_values_cpu(self):
        """Test scale computation for positive values on CPU."""
        tensor = np.array([[100.0, 200.0, 300.0]], dtype=np.float32)
        scale = compute_fp8_scale(tensor, backend="cpu")

        expected_scale = FP8Format.max_value / 300.0
        np.testing.assert_allclose(scale, expected_scale, rtol=1e-5)

    def test_negative_values_cpu(self):
        """Test scale computation for negative values on CPU."""
        tensor = np.array([[-100.0, -200.0, -300.0]], dtype=np.float32)
        scale = compute_fp8_scale(tensor, backend="cpu")

        # Should use absolute value
        expected_scale = FP8Format.max_value / 300.0
        np.testing.assert_allclose(scale, expected_scale, rtol=1e-5)

    def test_mixed_values_cpu(self):
        """Test scale computation for mixed values on CPU."""
        tensor = np.array([[100.0, -200.0, 300.0]], dtype=np.float32)
        scale = compute_fp8_scale(tensor, backend="cpu")

        expected_scale = FP8Format.max_value / 300.0
        np.testing.assert_allclose(scale, expected_scale, rtol=1e-5)

    def test_zero_tensor_cpu(self):
        """Test scale computation for zero tensor on CPU."""
        tensor = np.zeros((1024, 1024), dtype=np.float32)
        scale = compute_fp8_scale(tensor, backend="cpu")

        # Should return 1.0 for zero tensor
        assert scale == 1.0


class TestFP8Constants:
    """Test FP8 constants."""

    def test_fp8_max(self):
        """Test FP8 maximum value."""
        assert FP8Format.max_value == 448.0

    def test_fp8_range_coverage_cpu(self):
        """Test that FP8 covers expected range on CPU."""
        # Create tensor with FP8-range values
        tensor = np.random.uniform(-400, 400, (1024, 1024)).astype(np.float32)
        quantized, scale = quantize_fp8(tensor, backend="cpu")
        recovered = dequantize_fp8(quantized, scale, backend="cpu")

        # Relative error should be reasonable
        relative_error = np.abs(tensor - recovered) / (np.abs(tensor) + 1e-6)
        assert np.mean(relative_error) < 0.1
