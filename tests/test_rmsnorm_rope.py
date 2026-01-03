"""Property-based tests for fused RMSNorm + RoPE kernel.

Feature: triton-fused-operators
Tests Property 1: RMSNorm + RoPE Mathematical Correctness
Tests Property 3: Dimension Flexibility (RMSNorm + RoPE part)
Validates: Requirements 1.1, 1.2, 1.3, 1.4
"""

import pytest
import torch
from hypothesis import given, settings, strategies as st

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestRMSNormRoPECorrectness:
    """Property tests for RMSNorm + RoPE mathematical correctness."""
    
    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        seq_len=st.integers(min_value=1, max_value=512),
        hidden_dim=st.sampled_from([256, 512, 1024]),
        head_dim=st.sampled_from([32, 64]),
    )
    @settings(max_examples=100, deadline=None)
    def test_rmsnorm_rope_mathematical_correctness(
        self, batch_size, seq_len, hidden_dim, head_dim
    ):
        """
        Feature: triton-fused-operators, Property 1: RMSNorm + RoPE Mathematical Correctness
        Validates: Requirements 1.1, 1.2
        
        For any valid input tensor x, weight tensor w, and position embeddings (cos, sin),
        the fused RMSNorm + RoPE kernel output should be numerically equivalent to the
        sequential application of RMSNorm then RoPE.
        """
        from triton_ops.kernels.rmsnorm_rope import (
            fused_rmsnorm_rope,
            fused_rmsnorm_rope_reference,
        )
        
        # Ensure hidden_dim is divisible by head_dim
        if hidden_dim % head_dim != 0:
            hidden_dim = (hidden_dim // head_dim) * head_dim
        
        # Create inputs
        x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
        weight = torch.randn(hidden_dim, device='cuda', dtype=torch.float16)
        cos = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
        sin = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
        
        # Compute outputs
        triton_output = fused_rmsnorm_rope(x, weight, cos, sin)
        reference_output = fused_rmsnorm_rope_reference(x, weight, cos, sin)
        
        # Verify correctness
        assert torch.allclose(
            triton_output.float(),
            reference_output.float(),
            rtol=1e-3,
            atol=1e-5,
        ), f"Output mismatch: max diff = {(triton_output - reference_output).abs().max()}"
    
    @given(
        seq_len=st.integers(min_value=1, max_value=8192),
        hidden_dim=st.sampled_from([2048, 4096, 8192]),
    )
    @settings(max_examples=100, deadline=None)
    def test_dimension_flexibility(self, seq_len, hidden_dim):
        """
        Feature: triton-fused-operators, Property 3: Dimension Flexibility
        Validates: Requirements 1.3, 1.4
        
        For any valid combination of sequence length (1-8192) and supported hidden
        dimensions (2048, 4096, 8192), the kernel should produce correct outputs.
        """
        from triton_ops.kernels.rmsnorm_rope import (
            fused_rmsnorm_rope,
            fused_rmsnorm_rope_reference,
        )
        
        batch_size = 1
        head_dim = 64
        
        # Create inputs
        x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
        weight = torch.randn(hidden_dim, device='cuda', dtype=torch.float16)
        cos = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
        sin = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
        
        # Should not raise any errors
        triton_output = fused_rmsnorm_rope(x, weight, cos, sin)
        reference_output = fused_rmsnorm_rope_reference(x, weight, cos, sin)
        
        # Verify shape
        assert triton_output.shape == x.shape
        
        # Verify correctness
        assert torch.allclose(
            triton_output.float(),
            reference_output.float(),
            rtol=1e-3,
            atol=1e-5,
        )


class TestRMSNormRoPEEdgeCases:
    """Unit tests for edge cases."""
    
    def test_nan_propagation(self):
        """
        Validates: Requirement 1.7
        
        IF the input contains NaN values, THEN the kernel SHALL propagate them.
        """
        from triton_ops.kernels.rmsnorm_rope import fused_rmsnorm_rope
        
        batch_size, seq_len, hidden_dim, head_dim = 2, 16, 256, 64
        
        x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
        x[0, 0, 0] = float('nan')  # Inject NaN
        
        weight = torch.randn(hidden_dim, device='cuda', dtype=torch.float16)
        cos = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
        sin = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
        
        # Should not crash
        output = fused_rmsnorm_rope(x, weight, cos, sin)
        
        # NaN should propagate
        assert torch.isnan(output).any(), "NaN should propagate through the kernel"
    
    def test_inf_propagation(self):
        """
        Validates: Requirement 1.7
        
        IF the input contains Inf values, THEN the kernel SHALL propagate them.
        """
        from triton_ops.kernels.rmsnorm_rope import fused_rmsnorm_rope
        
        batch_size, seq_len, hidden_dim, head_dim = 2, 16, 256, 64
        
        x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
        x[0, 0, 0] = float('inf')  # Inject Inf
        
        weight = torch.randn(hidden_dim, device='cuda', dtype=torch.float16)
        cos = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
        sin = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
        
        # Should not crash
        output = fused_rmsnorm_rope(x, weight, cos, sin)
        
        # Inf should propagate (may become NaN due to operations)
        assert torch.isinf(output).any() or torch.isnan(output).any(), \
            "Inf should propagate through the kernel"
    
    def test_single_element(self):
        """Test with minimal dimensions."""
        from triton_ops.kernels.rmsnorm_rope import (
            fused_rmsnorm_rope,
            fused_rmsnorm_rope_reference,
        )
        
        batch_size, seq_len, hidden_dim, head_dim = 1, 1, 64, 64
        
        x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
        weight = torch.randn(hidden_dim, device='cuda', dtype=torch.float16)
        cos = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
        sin = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
        
        triton_output = fused_rmsnorm_rope(x, weight, cos, sin)
        reference_output = fused_rmsnorm_rope_reference(x, weight, cos, sin)
        
        assert torch.allclose(triton_output.float(), reference_output.float(), rtol=1e-3, atol=1e-5)
