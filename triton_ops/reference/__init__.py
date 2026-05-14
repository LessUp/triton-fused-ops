"""Reference implementations for Triton fused operators.

This module provides CPU (NumPy) and GPU (PyTorch) reference implementations
for all kernel families in the library. The unified interface allows:

1. **Correctness verification**: Compare Triton kernels against GPU reference
2. **CPU testing**: Run tests without GPU hardware
3. **Single source of truth**: One module for all reference implementations

Example:
    >>> from triton_ops.reference import fused_rmsnorm_rope, gated_mlp, quantize_fp8
    >>>
    >>> # CPU testing with NumPy
    >>> import numpy as np
    >>> x = np.random.randn(2, 128, 4096).astype(np.float32)
    >>> weight = np.ones(4096, dtype=np.float32)
    >>> output = fused_rmsnorm_rope(x, weight, cos, sin, backend='cpu')
    >>>
    >>> # GPU correctness verification with PyTorch
    >>> import torch
    >>> x_cuda = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
    >>> weight_cuda = torch.ones(4096, device='cuda', dtype=torch.float16)
    >>> output_cuda = fused_rmsnorm_rope(x_cuda, weight_cuda, cos_cuda, sin_cuda, backend='cuda')
"""

from triton_ops.reference.base import (
    Backend,
    BackendDispatcher,
    ensure_numpy,
    ensure_torch,
    reference_impl,
    to_output_dtype,
    validate_backend,
)
from triton_ops.reference.fp8 import dequantize_fp8, fp8_gemm, quantize_fp8
from triton_ops.reference.gated_mlp import gated_mlp
from triton_ops.reference.rmsnorm_rope import (
    fused_rmsnorm_rope,
    rmsnorm,
    rope,
)

__all__ = [
    # RMSNorm + RoPE
    "rmsnorm",
    "rope",
    "fused_rmsnorm_rope",
    # Gated MLP
    "gated_mlp",
    # FP8
    "quantize_fp8",
    "dequantize_fp8",
    "fp8_gemm",
    # Backend utilities
    "Backend",
    "validate_backend",
    "ensure_numpy",
    "ensure_torch",
    "to_output_dtype",
    "reference_impl",
    "BackendDispatcher",
]
