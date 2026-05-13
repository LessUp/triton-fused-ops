"""CPU-only tests for RMSNorm reference implementation."""

import numpy as np
import pytest

from triton_ops.reference import rmsnorm


class TestReferenceRMSNorm:
    """Test RMSNorm reference implementation."""

    def test_basic_normalization_cpu(self):
        """Test basic RMSNorm computation on CPU."""
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        weight = np.ones(4, dtype=np.float32)

        output = rmsnorm(x, weight, eps=1e-6, backend="cpu")

        # Verify shape
        assert output.shape == x.shape
        # Verify normalization effect
        assert not np.allclose(output, x)

    def test_weight_scaling_cpu(self):
        """Test that weight properly scales output on CPU."""
        x = np.random.randn(2, 128, 4096).astype(np.float32)
        weight = np.ones(4096, dtype=np.float32) * 2.0

        output = rmsnorm(x, weight, backend="cpu")
        output_half_weight = rmsnorm(x, weight / 2, backend="cpu")

        # Weight halved, output should be halved
        np.testing.assert_allclose(output / 2, output_half_weight, rtol=1e-5)

    @pytest.mark.parametrize("hidden_dim", [256, 512, 1024, 4096])
    def test_different_dimensions_cpu(self, hidden_dim):
        """Test RMSNorm with different hidden dimensions on CPU."""
        x = np.random.randn(2, 128, hidden_dim).astype(np.float32)
        weight = np.ones(hidden_dim, dtype=np.float32)

        output = rmsnorm(x, weight, backend="cpu")

        assert output.shape == x.shape

    def test_numerical_stability_small_cpu(self):
        """Test RMSNorm with very small values on CPU."""
        x = np.array([[1e-10, 1e-10, 1e-10]], dtype=np.float32)
        weight = np.ones(3, dtype=np.float32)

        # Should not produce NaN or Inf
        output = rmsnorm(x, weight, backend="cpu")
        assert np.all(np.isfinite(output))

    def test_numerical_stability_large_cpu(self):
        """Test RMSNorm with large values on CPU."""
        x = np.array([[1e6, 1e6, 1e6]], dtype=np.float32)
        weight = np.ones(3, dtype=np.float32)

        # Should not produce NaN or Inf
        output = rmsnorm(x, weight, backend="cpu")
        assert np.all(np.isfinite(output))

    def test_zero_input_cpu(self):
        """Test RMSNorm with zero input on CPU."""
        x = np.zeros((2, 128, 4096), dtype=np.float32)
        weight = np.ones(4096, dtype=np.float32)

        output = rmsnorm(x, weight, eps=1e-6, backend="cpu")

        # Output should be zero for zero input
        np.testing.assert_allclose(output, 0.0, atol=1e-7)

    def test_batch_independence_cpu(self):
        """Test that each batch element is normalized independently on CPU."""
        x = np.random.randn(4, 128, 4096).astype(np.float32)
        weight = np.ones(4096, dtype=np.float32)

        output = rmsnorm(x, weight, backend="cpu")

        # Compute RMSNorm for each batch independently
        for i in range(4):
            expected = rmsnorm(x[i : i + 1], weight, backend="cpu")
            np.testing.assert_allclose(output[i : i + 1], expected, rtol=1e-5)

    def test_different_shapes_cpu(self):
        """Test RMSNorm with different input shapes on CPU."""
        # 2D input
        x_2d = np.random.randn(128, 4096).astype(np.float32)
        weight = np.ones(4096, dtype=np.float32)
        output_2d = rmsnorm(x_2d, weight, backend="cpu")
        assert output_2d.shape == x_2d.shape

        # 3D input
        x_3d = np.random.randn(2, 128, 4096).astype(np.float32)
        output_3d = rmsnorm(x_3d, weight, backend="cpu")
        assert output_3d.shape == x_3d.shape
