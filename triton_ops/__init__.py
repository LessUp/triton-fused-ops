"""
Triton Fused Operators Library

High-performance Triton operators for Transformer models with operator fusion and FP8 quantization.

This library provides:
- Fused RMSNorm + RoPE kernel for efficient attention preprocessing
- Fused Gated MLP kernel for efficient feed-forward computation
- FP8 quantization and GEMM for reduced memory footprint
- Auto-tuning framework for optimal kernel configuration
- Benchmark suite for performance validation

Example usage:
    >>> import torch
    >>> from triton_ops import fused_rmsnorm_rope, fused_gated_mlp, fp8_gemm
    >>> 
    >>> # Fused RMSNorm + RoPE
    >>> x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
    >>> weight = torch.ones(4096, device='cuda', dtype=torch.float16)
    >>> cos = torch.randn(128, 64, device='cuda', dtype=torch.float16)
    >>> sin = torch.randn(128, 64, device='cuda', dtype=torch.float16)
    >>> output = fused_rmsnorm_rope(x, weight, cos, sin)
"""

__version__ = "0.1.0"

# Core functional API
from triton_ops.kernels.rmsnorm_rope import fused_rmsnorm_rope, FusedRMSNormRoPE
from triton_ops.kernels.gated_mlp import fused_gated_mlp, FusedGatedMLP
from triton_ops.kernels.fp8_gemm import fp8_gemm, FP8Linear
from triton_ops.kernels.fp8_quantize import quantize_fp8, dequantize_fp8

from triton_ops.models import (
    TensorSpec,
    RMSNormRoPEInput,
    GatedMLPInput,
    FP8GEMMInput,
    KernelMetrics,
    TuningResult,
    FP8Format,
)

from triton_ops.exceptions import (
    TritonKernelError,
    ShapeMismatchError,
    UnsupportedDtypeError,
    NumericalOverflowError,
    TuningFailedError,
)

# Auto-tuning framework
from triton_ops.autotuner import (
    TritonAutoTuner,
    ConfigCache,
    RMSNORM_ROPE_CONFIGS,
    GATED_MLP_CONFIGS,
    FP8_GEMM_CONFIGS,
)

# Benchmark suite
from triton_ops.benchmark import (
    BenchmarkSuite,
    CorrectnessVerifier,
    PerformanceReport,
)

__all__ = [
    # Fused kernels
    "fused_rmsnorm_rope",
    "FusedRMSNormRoPE",
    "fused_gated_mlp",
    "FusedGatedMLP",
    "fp8_gemm",
    "FP8Linear",
    "quantize_fp8",
    "dequantize_fp8",
    # Data models
    "TensorSpec",
    "RMSNormRoPEInput",
    "GatedMLPInput",
    "FP8GEMMInput",
    "KernelMetrics",
    "TuningResult",
    "FP8Format",
    # Exceptions
    "TritonKernelError",
    "ShapeMismatchError",
    "UnsupportedDtypeError",
    "NumericalOverflowError",
    "TuningFailedError",
    # Auto-tuning
    "TritonAutoTuner",
    "ConfigCache",
    "RMSNORM_ROPE_CONFIGS",
    "GATED_MLP_CONFIGS",
    "FP8_GEMM_CONFIGS",
    # Benchmark
    "BenchmarkSuite",
    "CorrectnessVerifier",
    "PerformanceReport",
]
