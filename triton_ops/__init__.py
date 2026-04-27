"""
Triton Fused Operators Library
==============================

High-performance Triton operators for Transformer models with operator fusion and FP8 quantization.

This library provides optimized GPU kernels for common Transformer operations:

**Core Kernels:**
    - :func:`fused_rmsnorm_rope` — Fused RMSNorm + Rotary Position Embedding
    - :func:`fused_gated_mlp` — Fused Gated MLP (SwiGLU/GeGLU)
    - :func:`fp8_gemm` — FP8 quantized matrix multiplication
    - :func:`quantize_fp8` / :func:`dequantize_fp8` — FP8 quantization utilities

**Module APIs:**
    - :class:`FusedRMSNormRoPE` — nn.Module wrapper for fused RMSNorm + RoPE
    - :class:`FusedGatedMLP` — nn.Module wrapper for fused Gated MLP
    - :class:`FP8Linear` — nn.Module for FP8 quantized linear layers

**Auto-Tuning:**
    - :class:`TritonAutoTuner` — Automatic kernel configuration optimization
    - :class:`ConfigCache` — Persistent cache for tuning results
    - Pre-defined config spaces: ``RMSNORM_ROPE_CONFIGS``, ``GATED_MLP_CONFIGS``, ``FP8_GEMM_CONFIGS``

**Benchmarking:**
    - :class:`BenchmarkSuite` — Comprehensive benchmark orchestration
    - :class:`CorrectnessVerifier` — Numerical correctness validation
    - :class:`PerformanceReport` — Performance metrics reporting

**Data Models:**
    - :class:`TensorSpec` — Tensor specification for validation
    - :class:`KernelMetrics` — Performance metrics container
    - :class:`TuningResult` — Auto-tuning result container
    - :class:`FP8Format` — FP8 format specification

Quick Start
-----------
>>> import torch
>>> from triton_ops import fused_rmsnorm_rope, fused_gated_mlp, fp8_gemm
>>>
>>> # Fused RMSNorm + RoPE
>>> x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
>>> weight = torch.ones(4096, device='cuda', dtype=torch.float16)
>>> cos = torch.randn(128, 64, device='cuda', dtype=torch.float16)
>>> sin = torch.randn(128, 64, device='cuda', dtype=torch.float16)
>>> output = fused_rmsnorm_rope(x, weight, cos, sin)
>>>
>>> # Fused Gated MLP (SwiGLU)
>>> gate_w = torch.randn(11008, 4096, device='cuda', dtype=torch.float16)
>>> up_w = torch.randn(11008, 4096, device='cuda', dtype=torch.float16)
>>> mlp_out = fused_gated_mlp(x, gate_w, up_w, activation='silu')
>>>
>>> # FP8 GEMM with automatic quantization
>>> a = torch.randn(1024, 4096, device='cuda', dtype=torch.float16)
>>> b = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
>>> gemm_out = fp8_gemm(a, b)  # Auto-quantizes inputs

Module API Example
------------------
>>> import torch
>>> from triton_ops import FusedRMSNormRoPE, FusedGatedMLP, FP8Linear
>>>
>>> class TransformerBlock(torch.nn.Module):
...     def __init__(self, hidden_dim=4096, head_dim=64, intermediate_dim=11008):
...         super().__init__()
...         self.norm = FusedRMSNormRoPE(hidden_dim, head_dim)
...         self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim, activation='silu')
...         self.proj = FP8Linear(intermediate_dim, hidden_dim)
...
...     def forward(self, x, cos, sin):
...         x = self.norm(x, cos, sin)
...         x = self.mlp(x)
...         return self.proj(x)

Performance Characteristics
---------------------------
+------------------------+------------+----------------------+
| Kernel                 | Speedup    | Memory Savings       |
+========================+============+======================+
| fused_rmsnorm_rope     | ~3x        | 50% HBM writes       |
| fused_gated_mlp        | ~1.5x      | 1 intermediate less |
| fp8_gemm               | ~1.4x      | 50% weight storage   |
+------------------------+------------+----------------------+

Hardware Requirements
---------------------
- **GPU:** NVIDIA Ampere (A100, RTX 30xx) or newer recommended
- **CUDA:** Version 11.8 or higher
- **Python:** 3.9 or higher
- **PyTorch:** 2.0 or higher
- **Triton:** 2.1 or higher

See Also
--------
- :mod:`triton_ops.kernels` — Low-level kernel implementations
- :mod:`triton_ops.autotuner` — Auto-tuning framework
- :mod:`triton_ops.benchmark` — Benchmarking utilities
- :mod:`triton_ops.validation` — Input validation utilities

References
----------
- `OpenAI Triton <https://github.com/openai/triton>`_
- `FlashAttention <https://github.com/Dao-AILab/flash-attention>`_
- `FP8 Formats for Deep Learning <https://arxiv.org/abs/2209.05433>`_
"""

__version__ = "1.0.1"

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
    DeviceError,
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
    "DeviceError",
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
