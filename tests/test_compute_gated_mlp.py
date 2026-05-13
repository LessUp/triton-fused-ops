"""CPU-only tests for Gated MLP reference implementation."""

import numpy as np
import pytest

from triton_ops.reference import gated_mlp
from triton_ops.reference.gated_mlp import _gelu_cpu, _silu_cpu


class TestActivationFunctions:
    """Test activation functions."""

    def test_silu_basic(self):
        """Test SiLU activation."""
        x = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
        output = _silu_cpu(x)

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
        output = _gelu_cpu(x)

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

        silu_out = _silu_cpu(x)
        gelu_out = _gelu_cpu(x)

        # Check for NaN/Inf
        assert np.all(np.isfinite(silu_out))
        assert np.all(np.isfinite(gelu_out))

        # Check for smoothness (no large jumps)
        silu_diff = np.diff(silu_out)
        gelu_diff = np.diff(gelu_out)

        # Maximum derivative should be reasonable
        assert np.abs(silu_diff).max() < 2.0
        assert np.abs(gelu_diff).max() < 2.0


class TestReferenceGatedMLP:
    """Test Gated MLP reference implementation."""

    def test_basic_computation_cpu(self):
        """Test basic Gated MLP computation on CPU."""
        batch, seq_len, hidden_dim = 2, 128, 4096
        intermediate_dim = 11264

        x = np.random.randn(batch, seq_len, hidden_dim).astype(np.float32) * 0.1
        gate_w = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.01
        up_w = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.01

        output = gated_mlp(x, gate_w, up_w, activation="silu", backend="cpu")

        # Verify shape
        assert output.shape == (batch, seq_len, intermediate_dim)

    def test_activation_variants_cpu(self):
        """Test Gated MLP with different activations on CPU."""
        x = np.random.randn(2, 128, 4096).astype(np.float32) * 0.1
        gate_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01
        up_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01

        output_silu = gated_mlp(x, gate_w, up_w, activation="silu", backend="cpu")
        output_gelu = gated_mlp(x, gate_w, up_w, activation="gelu", backend="cpu")

        # Different activations should produce different outputs
        assert not np.allclose(output_silu, output_gelu)

    def test_dimension_flexibility_cpu(self):
        """Test Gated MLP with different dimensions on CPU."""
        test_cases = [
            (4096, 11264),  # LLaMA-style
            (5120, 13824),  # Larger model
            (2048, 5632),  # Smaller model
        ]

        for hidden_dim, intermediate_dim in test_cases:
            x = np.random.randn(2, 128, hidden_dim).astype(np.float32) * 0.1
            gate_w = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.01
            up_w = np.random.randn(intermediate_dim, hidden_dim).astype(np.float32) * 0.01

            output = gated_mlp(x, gate_w, up_w, backend="cpu")

            assert output.shape == (2, 128, intermediate_dim)

    def test_weight_independence_cpu(self):
        """Test that gate and up weights are independent on CPU."""
        x = np.random.randn(1, 1, 4096).astype(np.float32) * 0.1
        gate_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01

        # Same weight for both
        output_same = gated_mlp(x, gate_w, gate_w, activation="silu", backend="cpu")

        # Different weights
        up_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01
        output_diff = gated_mlp(x, gate_w, up_w, activation="silu", backend="cpu")

        # Should produce different outputs
        assert not np.allclose(output_same, output_diff)

    def test_batch_independence_cpu(self):
        """Test that each batch element is processed independently on CPU."""
        x = np.random.randn(4, 128, 4096).astype(np.float32) * 0.1
        gate_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01
        up_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01

        output = gated_mlp(x, gate_w, up_w, backend="cpu")

        # Compute for each batch independently
        for i in range(4):
            expected = gated_mlp(x[i : i + 1], gate_w, up_w, backend="cpu")
            # Use larger tolerance for numerical precision
            np.testing.assert_allclose(output[i : i + 1], expected, rtol=1e-4, atol=1e-7)

    def test_zero_input_cpu(self):
        """Test Gated MLP with zero input on CPU."""
        x = np.zeros((2, 128, 4096), dtype=np.float32)
        gate_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01
        up_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01

        output = gated_mlp(x, gate_w, up_w, backend="cpu")

        # Zero input should produce zero output (both projections are zero)
        np.testing.assert_allclose(output, 0.0, atol=1e-7)

    def test_invalid_activation(self):
        """Test that invalid activation raises error."""
        x = np.random.randn(2, 128, 4096).astype(np.float32)
        gate_w = np.random.randn(11264, 4096).astype(np.float32)
        up_w = np.random.randn(11264, 4096).astype(np.float32)

        with pytest.raises(ValueError, match="activation must be"):
            gated_mlp(x, gate_w, up_w, activation="invalid", backend="cpu")
