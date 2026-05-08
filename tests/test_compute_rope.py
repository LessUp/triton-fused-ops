"""CPU-only tests for RoPE compute logic."""

import numpy as np
import pytest

from triton_ops.compute.rope import (
    compute_rope,
    compute_rope_frequencies,
    compute_rope_single_head,
)


class TestComputeRoPE:
    """Test RoPE computation logic."""

    def test_basic_rotation(self):
        """Test basic RoPE rotation."""
        x = np.random.randn(2, 128, 4096).astype(np.float32)
        cos = np.random.randn(128, 64).astype(np.float32)
        sin = np.random.randn(128, 64).astype(np.float32)

        output = compute_rope(x, cos, sin)

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
    def test_different_dimensions(self, hidden_dim, head_dim):
        """Test RoPE with different dimensions."""
        x = np.random.randn(2, 128, hidden_dim).astype(np.float32)
        cos = np.random.randn(128, head_dim).astype(np.float32)
        sin = np.random.randn(128, head_dim).astype(np.float32)

        output = compute_rope(x, cos, sin)

        assert output.shape == x.shape

    def test_position_embedding(self):
        """Test that different positions get different embeddings."""
        x = np.ones((1, 128, 4096), dtype=np.float32)
        cos, sin = compute_rope_frequencies(128, 64)

        output = compute_rope(x, cos, sin)

        # Each position should have different output
        for i in range(1, 128):
            assert not np.allclose(output[0, 0], output[0, i])

    def test_2d_input(self):
        """Test RoPE with 2D input (no batch dimension)."""
        x = np.random.randn(128, 4096).astype(np.float32)
        cos = np.random.randn(128, 64).astype(np.float32)
        sin = np.random.randn(128, 64).astype(np.float32)

        output = compute_rope(x, cos, sin)

        assert output.shape == x.shape

    def test_single_head_function(self):
        """Test single-head RoPE function."""
        head_dim = 64
        x = np.random.randn(head_dim).astype(np.float32)
        cos = np.random.randn(head_dim // 2).astype(np.float32)
        sin = np.random.randn(head_dim // 2).astype(np.float32)

        output = compute_rope_single_head(x, cos, sin)

        assert output.shape == x.shape

    def test_frequency_computation(self):
        """Test RoPE frequency computation."""
        seq_len = 128
        head_dim = 64

        cos, sin = compute_rope_frequencies(seq_len, head_dim)

        # Check shapes
        assert cos.shape == (seq_len, head_dim)
        assert sin.shape == (seq_len, head_dim)

        # Check values are in valid range
        assert np.all(cos >= -1.0) and np.all(cos <= 1.0)
        assert np.all(sin >= -1.0) and np.all(sin <= 1.0)

        # Check that different positions have different frequencies
        assert not np.allclose(cos[0], cos[1])
        assert not np.allclose(sin[0], sin[1])

    def test_rotation_matrix_property(self):
        """Test that RoPE rotation preserves norm."""
        x = np.random.randn(2, 128, 4096).astype(np.float32)
        cos, sin = compute_rope_frequencies(128, 64)

        output = compute_rope(x, cos, sin)

        # Norm should be approximately preserved (not exact due to splitting)
        x_norm = np.linalg.norm(x, axis=-1)
        output_norm = np.linalg.norm(output, axis=-1)

        np.testing.assert_allclose(x_norm, output_norm, rtol=1e-5)

    def test_reversibility_approximate(self):
        """Test that RoPE is approximately reversible (with inverse rotation)."""
        x = np.random.randn(1, 128, 4096).astype(np.float32)
        cos, sin = compute_rope_frequencies(128, 64)

        # Apply RoPE
        rotated = compute_rope(x, cos, sin)

        # Apply inverse rotation (use -sin)
        recovered = compute_rope(rotated, cos, -sin)

        # Should approximately recover original (with reasonable tolerance)
        np.testing.assert_allclose(x, recovered, rtol=1e-4, atol=1e-6)

    def test_batch_independence(self):
        """Test that each batch element is rotated independently."""
        x = np.random.randn(4, 128, 4096).astype(np.float32)
        cos = np.random.randn(128, 64).astype(np.float32)
        sin = np.random.randn(128, 64).astype(np.float32)

        output = compute_rope(x, cos, sin)

        # Compute RoPE for each batch independently
        for i in range(4):
            expected = compute_rope(x[i : i + 1], cos, sin)
            np.testing.assert_allclose(output[i : i + 1], expected, rtol=1e-5)

    def test_different_base_frequencies(self):
        """Test RoPE with different base frequencies."""
        seq_len = 128
        head_dim = 64

        cos1, sin1 = compute_rope_frequencies(seq_len, head_dim, base=10000.0)
        cos2, sin2 = compute_rope_frequencies(seq_len, head_dim, base=100000.0)

        # Different base should produce different frequencies
        assert not np.allclose(cos1, cos2)
        assert not np.allclose(sin1, sin2)
