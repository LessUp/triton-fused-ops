"""Property-based tests for fused Gated MLP kernel.

Feature: triton-fused-operators
Tests Property 2: Gated MLP Correctness with Activation Functions
Tests Property 3: Dimension Flexibility (Gated MLP part)
Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TestGatedMLPCorrectness:
    """Property tests for Gated MLP correctness."""

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=1, max_value=256),
        hidden_dim=st.sampled_from([256, 512, 1024]),
        intermediate_dim=st.sampled_from([512, 1024, 2048]),
        activation=st.sampled_from(["silu", "gelu"]),
    )
    @settings(max_examples=100, deadline=None)
    def test_gated_mlp_correctness(
        self, batch_size, seq_len, hidden_dim, intermediate_dim, activation
    ):
        """
        Feature: triton-fused-operators, Property 2: Gated MLP Correctness
        Validates: Requirements 2.1, 2.2, 2.3

        For any valid input tensor x, gate weights, up weights, and activation function,
        the fused Gated MLP kernel output should be numerically equivalent to:
        output = gate_proj(x) * activation(up_proj(x))
        """
        from triton_ops.kernels.gated_mlp import fused_gated_mlp, gated_mlp_reference

        # Create inputs
        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
        gate_weight = (
            torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16) * 0.1
        )
        up_weight = (
            torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16) * 0.1
        )

        # Compute outputs
        triton_output = fused_gated_mlp(x, gate_weight, up_weight, activation)
        reference_output = gated_mlp_reference(x, gate_weight, up_weight, activation)

        # Verify correctness
        assert torch.allclose(
            triton_output.float(),
            reference_output.float(),
            rtol=1e-2,
            atol=1e-4,
        ), f"Output mismatch for {activation}: max diff = {(triton_output - reference_output).abs().max()}"

    @given(
        batch_size=st.integers(min_value=1, max_value=64),
        intermediate_dim=st.sampled_from([5632, 11264, 22528]),
    )
    @settings(max_examples=50, deadline=None)
    def test_dimension_flexibility(self, batch_size, intermediate_dim):
        """
        Feature: triton-fused-operators, Property 3: Dimension Flexibility
        Validates: Requirements 2.4, 2.5

        For any valid batch size (1-64) and supported intermediate dimensions,
        the kernel should produce correct outputs.
        """
        from triton_ops.kernels.gated_mlp import fused_gated_mlp, gated_mlp_reference

        seq_len = 16
        hidden_dim = 4096

        # Create inputs
        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
        gate_weight = (
            torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16) * 0.1
        )
        up_weight = (
            torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16) * 0.1
        )

        # Should not raise any errors
        triton_output = fused_gated_mlp(x, gate_weight, up_weight, "silu")
        reference_output = gated_mlp_reference(x, gate_weight, up_weight, "silu")

        # Verify shape
        assert triton_output.shape == (batch_size, seq_len, intermediate_dim)

        # Verify correctness
        assert torch.allclose(
            triton_output.float(),
            reference_output.float(),
            rtol=1e-2,
            atol=1e-4,
        )


class TestGatedMLPActivations:
    """Unit tests for specific activation functions."""

    def test_silu_activation(self):
        """
        Validates: Requirement 2.2

        THE Fused_Operator SHALL support SiLU (Swish) activation function.
        """
        from triton_ops.kernels.gated_mlp import fused_gated_mlp, gated_mlp_reference

        batch_size, seq_len, hidden_dim, intermediate_dim = 2, 32, 512, 1024

        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
        gate_weight = (
            torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16) * 0.1
        )
        up_weight = (
            torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16) * 0.1
        )

        triton_output = fused_gated_mlp(x, gate_weight, up_weight, "silu")
        reference_output = gated_mlp_reference(x, gate_weight, up_weight, "silu")

        assert torch.allclose(triton_output.float(), reference_output.float(), rtol=1e-2, atol=1e-4)

    def test_gelu_activation(self):
        """
        Validates: Requirement 2.3

        THE Fused_Operator SHALL support GELU activation function.
        """
        from triton_ops.kernels.gated_mlp import fused_gated_mlp, gated_mlp_reference

        batch_size, seq_len, hidden_dim, intermediate_dim = 2, 32, 512, 1024

        x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
        gate_weight = (
            torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16) * 0.1
        )
        up_weight = (
            torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16) * 0.1
        )

        triton_output = fused_gated_mlp(x, gate_weight, up_weight, "gelu")
        reference_output = gated_mlp_reference(x, gate_weight, up_weight, "gelu")

        assert torch.allclose(triton_output.float(), reference_output.float(), rtol=1e-2, atol=1e-4)
