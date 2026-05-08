"""CPU-only tests for RMSNorm compute logic."""

import numpy as np
import pytest

from triton_ops.compute.rmsnorm import (
    compute_rms_variance,
    compute_rmsnorm,
    compute_rmsnorm_row,
)


class TestComputeRMSNorm:
    """Test RMSNorm computation logic."""

    def test_basic_normalization(self):
        """Test basic RMSNorm computation."""
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        weight = np.ones(4, dtype=np.float32)

        output = compute_rmsnorm(x, weight, eps=1e-6)

        # Verify shape
        assert output.shape == x.shape
        # Verify normalization effect
        assert not np.allclose(output, x)

    def test_weight_scaling(self):
        """Test that weight properly scales output."""
        x = np.random.randn(2, 128, 4096).astype(np.float32)
        weight = np.ones(4096, dtype=np.float32) * 2.0

        output = compute_rmsnorm(x, weight)
        output_half_weight = compute_rmsnorm(x, weight / 2)

        # Weight halved, output should be halved
        np.testing.assert_allclose(output / 2, output_half_weight, rtol=1e-5)

    @pytest.mark.parametrize("hidden_dim", [256, 512, 1024, 4096])
    def test_different_dimensions(self, hidden_dim):
        """Test RMSNorm with different hidden dimensions."""
        x = np.random.randn(2, 128, hidden_dim).astype(np.float32)
        weight = np.ones(hidden_dim, dtype=np.float32)

        output = compute_rmsnorm(x, weight)

        assert output.shape == x.shape

    def test_numerical_stability_small(self):
        """Test RMSNorm with very small values."""
        x = np.array([[1e-10, 1e-10, 1e-10]], dtype=np.float32)
        weight = np.ones(3, dtype=np.float32)

        # Should not produce NaN or Inf
        output = compute_rmsnorm(x, weight)
        assert np.all(np.isfinite(output))

    def test_numerical_stability_large(self):
        """Test RMSNorm with large values."""
        x = np.array([[1e6, 1e6, 1e6]], dtype=np.float32)
        weight = np.ones(3, dtype=np.float32)

        # Should not produce NaN or Inf
        output = compute_rmsnorm(x, weight)
        assert np.all(np.isfinite(output))

    def test_zero_input(self):
        """Test RMSNorm with zero input."""
        x = np.zeros((2, 128, 4096), dtype=np.float32)
        weight = np.ones(4096, dtype=np.float32)

        output = compute_rmsnorm(x, weight, eps=1e-6)

        # Output should be zero for zero input
        np.testing.assert_allclose(output, 0.0, atol=1e-7)

    def test_batch_independence(self):
        """Test that each batch element is normalized independently."""
        x = np.random.randn(4, 128, 4096).astype(np.float32)
        weight = np.ones(4096, dtype=np.float32)

        output = compute_rmsnorm(x, weight)

        # Compute RMSNorm for each batch independently
        for i in range(4):
            expected = compute_rmsnorm(x[i : i + 1], weight)
            np.testing.assert_allclose(output[i : i + 1], expected, rtol=1e-5)

    def test_row_function(self):
        """Test single-row RMSNorm function."""
        x_row = np.random.randn(4096).astype(np.float32)
        weight = np.ones(4096, dtype=np.float32)

        output = compute_rmsnorm_row(x_row, weight)

        # Verify output shape
        assert output.shape == x_row.shape

        # Verify consistency with batch version
        x_batch = x_row[np.newaxis, np.newaxis, :]
        output_batch = compute_rmsnorm(x_batch, weight)
        np.testing.assert_allclose(output, output_batch[0, 0], rtol=1e-5)

    def test_variance_computation(self):
        """Test variance computation helper."""
        x = np.random.randn(2, 128, 4096).astype(np.float32)

        variance = compute_rms_variance(x)

        # Should be positive
        assert variance > 0

        # Should match direct computation
        expected_variance = float(np.mean(x**2))
        np.testing.assert_allclose(variance, expected_variance, rtol=1e-5)

    def test_different_shapes(self):
        """Test RMSNorm with different input shapes."""
        # 2D input
        x_2d = np.random.randn(128, 4096).astype(np.float32)
        weight = np.ones(4096, dtype=np.float32)
        output_2d = compute_rmsnorm(x_2d, weight)
        assert output_2d.shape == x_2d.shape

        # 3D input
        x_3d = np.random.randn(2, 128, 4096).astype(np.float32)
        output_3d = compute_rmsnorm(x_3d, weight)
        assert output_3d.shape == x_3d.shape
