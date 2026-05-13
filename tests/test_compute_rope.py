"""CPU-only tests for RoPE reference implementation."""

import numpy as np
import pytest

from triton_ops.reference import rope
from triton_ops.reference.rmsnorm_rope import compute_rope_frequencies


class TestReferenceRoPE:
    """Test RoPE reference implementation."""

    def test_basic_rotation_cpu(self):
        """Test basic RoPE rotation on CPU."""
        x = np.random.randn(2, 128, 4096).astype(np.float32)
        cos = np.random.randn(128, 64).astype(np.float32)
        sin = np.random.randn(128, 64).astype(np.float32)

        output = rope(x, cos, sin, backend="cpu")

        # Verify shape
        assert output.shape == x.shape
        # Verify transformation occurred
        assert not np.allclose(output, x)

    @pytest.mark.parametrize(
        "hidden_dim,head_dim",
        [
            (4096, 64),
            (5120, 128),
            (2048, 32),
        ],
    )
    def test_different_dimensions_cpu(self, hidden_dim, head_dim):
        """Test RoPE with different dimensions on CPU."""
        x = np.random.randn(2, 128, hidden_dim).astype(np.float32)
        cos = np.random.randn(128, head_dim).astype(np.float32)
        sin = np.random.randn(128, head_dim).astype(np.float32)

        output = rope(x, cos, sin, backend="cpu")

        assert output.shape == x.shape

    def test_position_embedding_cpu(self):
        """Test that different positions get different embeddings on CPU."""
        x = np.ones((1, 128, 4096), dtype=np.float32)
        cos, sin = compute_rope_frequencies(128, 64, backend="cpu")

        output = rope(x, cos, sin, backend="cpu")

        # Each position should have different output
        for i in range(1, 128):
            assert not np.allclose(output[0, 0], output[0, i])

    def test_2d_input_cpu(self):
        """Test RoPE with 2D input (no batch dimension) on CPU."""
        x = np.random.randn(128, 4096).astype(np.float32)
        cos = np.random.randn(128, 64).astype(np.float32)
        sin = np.random.randn(128, 64).astype(np.float32)

        output = rope(x, cos, sin, backend="cpu")

        assert output.shape == x.shape

    def test_frequency_computation_cpu(self):
        """Test RoPE frequency computation on CPU."""
        seq_len = 128
        head_dim = 64

        cos, sin = compute_rope_frequencies(seq_len, head_dim, backend="cpu")

        # Check shapes
        assert cos.shape == (seq_len, head_dim)
        assert sin.shape == (seq_len, head_dim)

        # Check values are in valid range
        assert np.all(cos >= -1.0) and np.all(cos <= 1.0)
        assert np.all(sin >= -1.0) and np.all(sin <= 1.0)

        # Check that different positions have different frequencies
        assert not np.allclose(cos[0], cos[1])
        assert not np.allclose(sin[0], sin[1])

    def test_rotation_matrix_property_cpu(self):
        """Test that RoPE rotation preserves norm on CPU."""
        x = np.random.randn(2, 128, 4096).astype(np.float32)
        cos, sin = compute_rope_frequencies(128, 64, backend="cpu")

        output = rope(x, cos, sin, backend="cpu")

        # Norm should be approximately preserved (not exact due to splitting)
        x_norm = np.linalg.norm(x, axis=-1)
        output_norm = np.linalg.norm(output, axis=-1)

        np.testing.assert_allclose(x_norm, output_norm, rtol=1e-5)

    def test_reversibility_approximate_cpu(self):
        """Test that RoPE is approximately reversible on CPU."""
        x = np.random.randn(1, 128, 4096).astype(np.float32)
        cos, sin = compute_rope_frequencies(128, 64, backend="cpu")

        # Apply RoPE
        rotated = rope(x, cos, sin, backend="cpu")

        # Apply inverse rotation (use -sin)
        recovered = rope(rotated, cos, -sin, backend="cpu")

        # Should approximately recover original (with reasonable tolerance)
        np.testing.assert_allclose(x, recovered, rtol=1e-4, atol=1e-6)

    def test_batch_independence_cpu(self):
        """Test that each batch element is rotated independently on CPU."""
        x = np.random.randn(4, 128, 4096).astype(np.float32)
        cos = np.random.randn(128, 64).astype(np.float32)
        sin = np.random.randn(128, 64).astype(np.float32)

        output = rope(x, cos, sin, backend="cpu")

        # Compute RoPE for each batch independently
        for i in range(4):
            expected = rope(x[i : i + 1], cos, sin, backend="cpu")
            np.testing.assert_allclose(output[i : i + 1], expected, rtol=1e-5)

    def test_different_base_frequencies_cpu(self):
        """Test RoPE with different base frequencies on CPU."""
        seq_len = 128
        head_dim = 64

        cos1, sin1 = compute_rope_frequencies(seq_len, head_dim, base=10000.0, backend="cpu")
        cos2, sin2 = compute_rope_frequencies(seq_len, head_dim, base=100000.0, backend="cpu")

        # Different base should produce different frequencies
        assert not np.allclose(cos1, cos2)
        assert not np.allclose(sin1, sin2)
