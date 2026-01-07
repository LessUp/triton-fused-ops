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
# Auto-tuning framework
from triton_ops.autotuner import (
    FP8_GEMM_CONFIGS,
    GATED_MLP_CONFIGS,
    RMSNORM_ROPE_CONFIGS,
    ConfigCache,
    TritonAutoTuner,
)

# Benchmark suite
from triton_ops.benchmark import (
    BenchmarkSuite,
    CorrectnessVerifier,
    PerformanceReport,
)
from triton_ops.exceptions import (
    NumericalOverflowError,
    ShapeMismatchError,
    TritonKernelError,
    TuningFailedError,
    UnsupportedDtypeError,
)
from triton_ops.kernels.fp8_gemm import FP8Linear, fp8_gemm
from triton_ops.kernels.fp8_quantize import dequantize_fp8, quantize_fp8
from triton_ops.kernels.gated_mlp import FusedGatedMLP, fused_gated_mlp
from triton_ops.kernels.rmsnorm_rope import FusedRMSNormRoPE, fused_rmsnorm_rope
from triton_ops.models import (
    FP8Format,
    FP8GEMMInput,
    GatedMLPInput,
    KernelMetrics,
    RMSNormRoPEInput,
    TensorSpec,
    TuningResult,
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
