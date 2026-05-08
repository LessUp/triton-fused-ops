"""CPU-only tests for Gated MLP compute logic."""

import numpy as np
import pytest

from triton_ops.compute.gated_mlp import (
    compute_gated_mlp,
    compute_gated_mlp_single,
    gelu,
    silu,
)


class TestActivationFunctions:
    """Test activation functions."""

    def test_silu_basic(self):
        """Test SiLU activation."""
        x = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
        output = silu(x)

        # SiLU(0) = 0
        assert output[0] == 0.0
        # SiLU is always positive for positive input
        assert output[1] > 0
        assert output[3] > 0
        # SiLU is always negative for negative input
        assert output[2] < 0
        assert output[4] < 0

    def test_gelu_basic(self):
        """Test GELU activation."""
        x = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
        output = gelu(x)

        # GELU(0) ≈ 0
        assert abs(output[0]) < 1e-5
        # GELU is always positive for positive input
        assert output[1] > 0
        assert output[3] > 0
        # GELU can be negative for negative input
        assert output[2] < 0
        assert output[4] < 0

    def test_activation_smoothness(self):
        """Test that activations are smooth (no discontinuities)."""
        x = np.linspace(-5, 5, 1000, dtype=np.float32)

        silu_out = silu(x)
        gelu_out = gelu(x)

        # Check for NaN/Inf
        assert np.all(np.isfinite(silu_out))
        assert np.all(np.isfinite(gelu_out))

        # Check for smoothness (no large jumps)
        silu_diff = np.diff(silu_out)
        gelu_diff = np.diff(gelu_out)

        # Maximum derivative should be reasonable
        assert np.abs(silu_diff).max() < 2.0
        assert np.abs(gelu_diff).max() < 2.0


class TestComputeGatedMLP:
    """Test Gated MLP computation logic."""

    def test_basic_computation(self):
        """Test basic Gated MLP computation."""
        batch, seq_len, hidden_dim = 2, 128, 4096
        intermediate_dim = 11264

        x = np.random.randn(batch, seq_len, hidden_dim).astype(np.float32) * 0.1
        gate_w = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.01
        up_w = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.01

        output = compute_gated_mlp(x, gate_w, up_w, activation="silu")

        # Verify shape
        assert output.shape == (batch, seq_len, intermediate_dim)

    def test_activation_variants(self):
        """Test Gated MLP with different activations."""
        x = np.random.randn(2, 128, 4096).astype(np.float32) * 0.1
        gate_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01
        up_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01

        output_silu = compute_gated_mlp(x, gate_w, up_w, activation="silu")
        output_gelu = compute_gated_mlp(x, gate_w, up_w, activation="gelu")

        # Different activations should produce different outputs
        assert not np.allclose(output_silu, output_gelu)

    def test_dimension_flexibility(self):
        """Test Gated MLP with different dimensions."""
        test_cases = [
            (4096, 11264),  # LLaMA-style
            (5120, 13824),  # Larger model
            (2048, 5632),  # Smaller model
        ]

        for hidden_dim, intermediate_dim in test_cases:
            x = np.random.randn(2, 128, hidden_dim).astype(np.float32) * 0.1
            gate_w = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.01
            up_w = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.01

            output = compute_gated_mlp(x, gate_w, up_w)

            assert output.shape == (2, 128, intermediate_dim)

    def test_weight_independence(self):
        """Test that gate and up weights are independent."""
        x = np.random.randn(1, 1, 4096).astype(np.float32) * 0.1
        gate_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01

        # Same weight for both
        output_same = compute_gated_mlp(x, gate_w, gate_w, activation="silu")

        # Different weights
        up_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01
        output_diff = compute_gated_mlp(x, gate_w, up_w, activation="silu")

        # Should produce different outputs
        assert not np.allclose(output_same, output_diff)

    def test_batch_independence(self):
        """Test that each batch element is processed independently."""
        x = np.random.randn(4, 128, 4096).astype(np.float32) * 0.1
        gate_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01
        up_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01

        output = compute_gated_mlp(x, gate_w, up_w)

        # Compute for each batch independently
        for i in range(4):
            expected = compute_gated_mlp(x[i : i + 1], gate_w, up_w)
            # Use larger tolerance for numerical precision
            np.testing.assert_allclose(output[i : i + 1], expected, rtol=1e-4, atol=1e-7)

    def test_single_function(self):
        """Test single-position Gated MLP function."""
        hidden_dim = 4096
        intermediate_dim = 11264

        x = np.random.randn(hidden_dim).astype(np.float32) * 0.1
        gate_w = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.01
        up_w = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.01

        output = compute_gated_mlp_single(x, gate_w, up_w, activation="silu")

        assert output.shape == (intermediate_dim,)

        # Verify consistency with batch version
        x_batch = x[np.newaxis, np.newaxis, :]
        output_batch = compute_gated_mlp(x_batch, gate_w, up_w)
        np.testing.assert_allclose(output, output_batch[0, 0], rtol=1e-5)

    def test_zero_input(self):
        """Test Gated MLP with zero input."""
        x = np.zeros((2, 128, 4096), dtype=np.float32)
        gate_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01
        up_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01

        output = compute_gated_mlp(x, gate_w, up_w)

        # Zero input should produce zero output (both projections are zero)
        np.testing.assert_allclose(output, 0.0, atol=1e-7)

    def test_invalid_activation(self):
        """Test that invalid activation raises error."""
        x = np.random.randn(2, 128, 4096).astype(np.float32)
        gate_w = np.random.randn(11264, 4096).astype(np.float32)
        up_w = np.random.randn(11264, 4096).astype(np.float32)

        with pytest.raises(ValueError, match="Unknown activation"):
            compute_gated_mlp(x, gate_w, up_w, activation="invalid")
