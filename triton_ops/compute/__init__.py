"""CPU-testable computation logic for Triton operators.

This module provides pure NumPy implementations of the computational
logic used in Triton kernels. These functions can be tested independently
of GPU hardware, improving testability and enabling CPU-based verification.

Available functions:
    - compute_rmsnorm: RMSNorm normalization
    - compute_rope: Rotary Position Embedding
    - compute_gated_mlp: Gated MLP (SwiGLU/GeGLU)
    - compute_quantize_fp8: FP8 quantization
    - compute_dequantize_fp8: FP8 dequantization

These implementations serve as:
    1. Reference implementations for kernel correctness verification
    2. CPU-testable logic for unit testing
    3. Documentation of the mathematical operations

Note:
    All functions use NumPy arrays and can be tested without GPU.
"""

from triton_ops.compute.fp8 import (
    compute_dequantize_fp8,
    compute_quantize_fp8,
)
from triton_ops.compute.gated_mlp import compute_gated_mlp
from triton_ops.compute.rmsnorm import compute_rmsnorm
from triton_ops.compute.rope import compute_rope

__all__ = [
    "compute_rmsnorm",
    "compute_rope",
    "compute_gated_mlp",
    "compute_quantize_fp8",
    "compute_dequantize_fp8",
]
