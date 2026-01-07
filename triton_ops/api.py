"""Convenient API for Triton fused operators.

This module provides a unified, user-friendly interface for all Triton operators.
"""

from typing import Literal, Optional

import torch

from triton_ops.kernels.fp8_gemm import (
    FP8Linear,
)
from triton_ops.kernels.fp8_gemm import (
    fp8_gemm as _fp8_gemm,
)
from triton_ops.kernels.fp8_quantize import (
    dequantize_fp8 as _dequantize_fp8,
)
from triton_ops.kernels.fp8_quantize import (
    quantize_fp8 as _quantize_fp8,
)
from triton_ops.kernels.gated_mlp import (
    FusedGatedMLP,
)
from triton_ops.kernels.gated_mlp import (
    fused_gated_mlp as _fused_gated_mlp,
)
from triton_ops.kernels.rmsnorm_rope import (
    FusedRMSNormRoPE,
)
from triton_ops.kernels.rmsnorm_rope import (
    fused_rmsnorm_rope as _fused_rmsnorm_rope,
)


def fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
    num_heads: Optional[int] = None,
) -> torch.Tensor:
    """Apply fused RMSNorm + RoPE transformation.

    This function combines RMSNorm and Rotary Position Embedding into a single
    kernel launch, reducing memory bandwidth requirements by eliminating
    intermediate HBM writes.

    Mathematical operations:
    1. RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
    2. RoPE: y_rope = y * cos + rotate_half(y) * sin

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim]
        weight: RMSNorm weight of shape [hidden_dim]
        cos: Cosine position embeddings of shape [seq_len, head_dim]
        sin: Sine position embeddings of shape [seq_len, head_dim]
        eps: Small constant for numerical stability (default: 1e-6)
        num_heads: Number of attention heads (inferred from hidden_dim/head_dim if not provided)

    Returns:
        Output tensor of shape [batch, seq_len, hidden_dim] with RMSNorm + RoPE applied

    Example:
        >>> x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
        >>> weight = torch.ones(4096, device='cuda', dtype=torch.float16)
        >>> cos = torch.randn(128, 64, device='cuda', dtype=torch.float16)
        >>> sin = torch.randn(128, 64, device='cuda', dtype=torch.float16)
        >>> output = fused_rmsnorm_rope(x, weight, cos, sin)
    """
    return _fused_rmsnorm_rope(x, weight, cos, sin, eps, num_heads)


def fused_gated_mlp(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    activation: Literal["silu", "gelu"] = "silu",
) -> torch.Tensor:
    """Apply fused Gated MLP transformation.

    Computes: output = gate_proj(x) * activation(up_proj(x))

    This fused implementation reduces memory bandwidth by computing both
    projections and the activation in a single kernel.

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim]
        gate_weight: Gate projection weight of shape [intermediate_dim, hidden_dim]
        up_weight: Up projection weight of shape [intermediate_dim, hidden_dim]
        activation: Activation function - "silu" (default) or "gelu"

    Returns:
        Output tensor of shape [batch, seq_len, intermediate_dim]

    Example:
        >>> x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
        >>> gate_w = torch.randn(11264, 4096, device='cuda', dtype=torch.float16)
        >>> up_w = torch.randn(11264, 4096, device='cuda', dtype=torch.float16)
        >>> output = fused_gated_mlp(x, gate_w, up_w, activation="silu")
    """
    return _fused_gated_mlp(x, gate_weight, up_weight, activation)


def fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Perform FP8 matrix multiplication.

    If inputs are not already in FP8 format, they will be quantized automatically.
    Uses FP32 accumulation for numerical stability.

    Args:
        a: First matrix of shape [M, K] - can be FP8 (uint8) or float
        b: Second matrix of shape [K, N] - can be FP8 (uint8) or float
        a_scale: Scale factor for A (required if A is FP8, computed if float)
        b_scale: Scale factor for B (required if B is FP8, computed if float)
        output_dtype: Output data type - float16 (default) or bfloat16

    Returns:
        Result matrix of shape [M, N] in output_dtype

    Example:
        >>> a = torch.randn(512, 1024, device='cuda', dtype=torch.float16)
        >>> b = torch.randn(1024, 2048, device='cuda', dtype=torch.float16)
        >>> c = fp8_gemm(a, b)  # Auto-quantizes to FP8
    """
    return _fp8_gemm(a, b, a_scale, b_scale, output_dtype)


def quantize_fp8(
    tensor: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to FP8 E4M3 format.

    FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits.
    Max representable value: 448.0

    Args:
        tensor: Input tensor in FP16/BF16/FP32
        scale: Optional pre-computed scale factor. If None, computed automatically.

    Returns:
        Tuple of (quantized_tensor, scale_factor)
        - quantized_tensor: FP8 values stored as uint8
        - scale_factor: Scale used for quantization

    Example:
        >>> tensor = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
        >>> quantized, scale = quantize_fp8(tensor)
    """
    return _quantize_fp8(tensor, scale)


def dequantize_fp8(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize FP8 tensor back to FP16/BF16.

    Args:
        tensor: FP8 tensor stored as uint8
        scale: Scale factor used during quantization
        output_dtype: Output data type - float16 (default) or bfloat16

    Returns:
        Dequantized tensor in specified dtype

    Example:
        >>> quantized, scale = quantize_fp8(original_tensor)
        >>> recovered = dequantize_fp8(quantized, scale)
    """
    return _dequantize_fp8(tensor, scale, output_dtype)


# Re-export module classes for convenience
__all__ = [
    # Functional API
    "fused_rmsnorm_rope",
    "fused_gated_mlp",
    "fp8_gemm",
    "quantize_fp8",
    "dequantize_fp8",
    # Module classes
    "FusedRMSNormRoPE",
    "FusedGatedMLP",
    "FP8Linear",
]
