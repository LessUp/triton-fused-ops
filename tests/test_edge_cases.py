"""Unit tests for edge cases across all kernels.

This module contains unit tests for edge cases, error conditions,
and boundary values that are not covered by property-based tests.
"""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TestValidationErrors:
    """Tests for input validation and error handling."""

    def test_rmsnorm_rope_shape_mismatch(self):
        """Test that shape mismatches raise appropriate errors."""
        from triton_ops.exceptions import ShapeMismatchError
        from triton_ops.validation import validate_rmsnorm_rope_inputs

        x = torch.randn(2, 16, 256, device="cuda", dtype=torch.float16)
        weight = torch.randn(128, device="cuda", dtype=torch.float16)  # Wrong size
        cos = torch.randn(16, 64, device="cuda", dtype=torch.float16)
        sin = torch.randn(16, 64, device="cuda", dtype=torch.float16)

        with pytest.raises(ShapeMismatchError):
            validate_rmsnorm_rope_inputs(x, weight, cos, sin)

    def test_rmsnorm_rope_wrong_dtype(self):
        """Test that unsupported dtypes raise appropriate errors."""
        from triton_ops.exceptions import UnsupportedDtypeError
        from triton_ops.validation import validate_rmsnorm_rope_inputs

        x = torch.randn(2, 16, 256, device="cuda", dtype=torch.int32)  # Wrong dtype
        weight = torch.randn(256, device="cuda", dtype=torch.float16)
        cos = torch.randn(16, 64, device="cuda", dtype=torch.float16)
        sin = torch.randn(16, 64, device="cuda", dtype=torch.float16)

        with pytest.raises(UnsupportedDtypeError):
            validate_rmsnorm_rope_inputs(x, weight, cos, sin)

    def test_gated_mlp_invalid_activation(self):
        """Test that invalid activation raises error."""
        from triton_ops.validation import validate_gated_mlp_inputs

        x = torch.randn(2, 16, 256, device="cuda", dtype=torch.float16)
        gate_w = torch.randn(512, 256, device="cuda", dtype=torch.float16)
        up_w = torch.randn(512, 256, device="cuda", dtype=torch.float16)

        with pytest.raises(ValueError):
            validate_gated_mlp_inputs(x, gate_w, up_w, activation="invalid")

    def test_fp8_gemm_dimension_mismatch(self):
        """Test that matrix dimension mismatches raise errors."""
        from triton_ops.exceptions import ShapeMismatchError
        from triton_ops.validation import validate_fp8_gemm_inputs

        a = torch.randn(64, 128, device="cuda", dtype=torch.float16)
        b = torch.randn(64, 64, device="cuda", dtype=torch.float16)  # K mismatch
        a_scale = torch.tensor(1.0, device="cuda")
        b_scale = torch.tensor(1.0, device="cuda")

        with pytest.raises(ShapeMismatchError):
            validate_fp8_gemm_inputs(a, b, a_scale, b_scale)


class TestDataModels:
    """Tests for data model classes."""

    def test_tensor_spec_validation(self):
        """Test TensorSpec validation."""
        from triton_ops.models import TensorSpec

        spec = TensorSpec(
            shape=(2, 16, 256),
            dtype=torch.float16,
            device="cuda",
        )

        # Valid tensor
        valid = torch.randn(2, 16, 256, device="cuda", dtype=torch.float16)
        assert spec.validate(valid)

        # Wrong shape
        wrong_shape = torch.randn(2, 16, 128, device="cuda", dtype=torch.float16)
        assert not spec.validate(wrong_shape)

        # Wrong dtype
        wrong_dtype = torch.randn(2, 16, 256, device="cuda", dtype=torch.float32)
        assert not spec.validate(wrong_dtype)

    def test_fp8_format_scale_computation(self):
        """Test FP8Format scale computation."""
        from triton_ops.models import FP8Format

        # Normal tensor
        tensor = torch.randn(100, device="cuda", dtype=torch.float16) * 10
        scale = FP8Format.compute_scale(tensor)

        # Scale should map max value to FP8 max
        scaled_max = (tensor.abs().max() * scale).item()
        assert abs(scaled_max - FP8Format.max_value) < 1.0

        # Zero tensor
        zero_tensor = torch.zeros(100, device="cuda", dtype=torch.float16)
        zero_scale = FP8Format.compute_scale(zero_tensor)
        assert zero_scale.item() == 1.0

    def test_kernel_metrics_str(self):
        """Test KernelMetrics string representation."""
        from triton_ops.models import KernelMetrics

        metrics = KernelMetrics(
            latency_ms=1.5,
            throughput_tflops=10.0,
            bandwidth_gbps=1000.0,
            bandwidth_utilization=50.0,
        )

        str_repr = str(metrics)
        assert "1.5" in str_repr or "1.500" in str_repr
        assert "10" in str_repr
        assert "1000" in str_repr
        assert "50" in str_repr


class TestExceptionAttributes:
    """Tests for exception class attributes."""

    def test_shape_mismatch_error_attributes(self):
        """Test ShapeMismatchError attributes."""
        from triton_ops.exceptions import ShapeMismatchError

        error = ShapeMismatchError(
            "Shape mismatch",
            expected=(2, 16, 256),
            actual=(2, 16, 128),
            tensor_name="x",
        )

        assert error.expected == (2, 16, 256)
        assert error.actual == (2, 16, 128)
        assert error.tensor_name == "x"

    def test_numerical_overflow_error_attributes(self):
        """Test NumericalOverflowError attributes."""
        from triton_ops.exceptions import NumericalOverflowError

        error = NumericalOverflowError(
            "Overflow",
            max_value=1000.0,
            scale=0.5,
            attempts=3,
        )

        assert error.max_value == 1000.0
        assert error.scale == 0.5
        assert error.attempts == 3


class TestModuleWrappers:
    """Tests for PyTorch module wrappers."""

    def test_fused_rmsnorm_rope_module(self):
        """Test FusedRMSNormRoPE module."""
        from triton_ops.kernels.rmsnorm_rope import FusedRMSNormRoPE

        hidden_dim = 256
        head_dim = 64

        module = FusedRMSNormRoPE(hidden_dim, head_dim)

        # Check parameters
        assert module.weight.shape == (hidden_dim,)
        assert module.num_heads == hidden_dim // head_dim

        # Forward pass
        x = torch.randn(2, 16, hidden_dim, device="cuda", dtype=torch.float16)
        cos = torch.randn(16, head_dim, device="cuda", dtype=torch.float16)
        sin = torch.randn(16, head_dim, device="cuda", dtype=torch.float16)

        module = module.cuda()
        output = module(x, cos, sin)

        assert output.shape == x.shape

    def test_fused_gated_mlp_module(self):
        """Test FusedGatedMLP module."""
        from triton_ops.kernels.gated_mlp import FusedGatedMLP

        hidden_dim = 256
        intermediate_dim = 512

        module = FusedGatedMLP(hidden_dim, intermediate_dim, activation="silu")

        # Check parameters
        assert module.gate_weight.shape == (intermediate_dim, hidden_dim)
        assert module.up_weight.shape == (intermediate_dim, hidden_dim)

        # Forward pass
        x = torch.randn(2, 16, hidden_dim, device="cuda", dtype=torch.float16)

        module = module.cuda()
        output = module(x)

        assert output.shape == (2, 16, intermediate_dim)

    def test_fp8_linear_module(self):
        """Test FP8Linear module."""
        from triton_ops.kernels.fp8_gemm import FP8Linear

        in_features = 256
        out_features = 512

        module = FP8Linear(in_features, out_features)

        # Check parameters
        assert module.weight.shape == (out_features, in_features)

        # Forward pass
        x = torch.randn(2, 16, in_features, device="cuda", dtype=torch.float16)

        module = module.cuda()
        output = module(x)

        assert output.shape == (2, 16, out_features)
