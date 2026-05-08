"""Triton kernel implementations.

This module exports only the public APIs. Internal kernel functions
(*_kernel) are kept private and should not be used directly.
"""

from triton_ops.kernels.fp8_gemm import (
    FP8Linear,
    fp8_gemm,
)
from triton_ops.kernels.fp8_quantize import (
    dequantize_fp8,
    quantize_fp8,
)
from triton_ops.kernels.gated_mlp import (
    FusedGatedMLP,
    fused_gated_mlp,
)
from triton_ops.kernels.rmsnorm_rope import (
    FusedRMSNormRoPE,
    fused_rmsnorm_rope,
)

__all__ = [
    # Public functional APIs
    "fused_rmsnorm_rope",
    "fused_gated_mlp",
    "fp8_gemm",
    "quantize_fp8",
    "dequantize_fp8",
    # Public module APIs
    "FusedRMSNormRoPE",
    "FusedGatedMLP",
    "FP8Linear",
]
