"""FP8 quantization computation logic - CPU testable.

This module provides pure NumPy implementations of FP8 quantization
that can be tested independently of Triton kernels.

FP8 E4M3 format:
    - 1 sign bit, 4 exponent bits, 3 mantissa bits
    - Maximum representable value: 448.0
    - Minimum normal: 2^-6 ≈ 0.015625
"""

from typing import Any, Optional, Tuple

import numpy as np

# FP8 E4M3 constants
FP8_MAX = 448.0
FP8_MIN_NORMAL = 2**-6


def compute_quantize_fp8(
    tensor: np.ndarray,
    scale: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Quantize tensor to FP8 E4M3 format.

    Args:
        tensor: Input tensor in float32
        scale: Optional pre-computed scale factor. If None, computed automatically.

    Returns:
        Tuple of (quantized_tensor, scale_factor)
        - quantized_tensor: FP8 values stored as uint8 (for compatibility)
        - scale_factor: Scale used for quantization

    Example:
        >>> tensor = np.random.randn(1024, 1024).astype(np.float32) * 10
        >>> quantized, scale = compute_quantize_fp8(tensor)
        >>> quantized.dtype
        dtype('uint8')
        >>> scale > 0
        True

    Note:
        This is a simplified implementation for CPU testing.
        Actual FP8 hardware uses more sophisticated rounding.
    """
    # Compute scale if not provided
    if scale is None:
        max_abs = np.abs(tensor).max()
        if max_abs == 0:
            scale = 1.0
        else:
            scale = FP8_MAX / max_abs

    # Scale the tensor
    scaled = tensor * scale

    # Clip to FP8 range
    scaled = np.clip(scaled, -FP8_MAX, FP8_MAX)

    # Convert to FP8-like representation
    # This is a simplified version - real FP8 uses proper rounding
    # Map [-FP8_MAX, FP8_MAX] to [0, 255]
    # FP8 values are stored as: value = (uint8 - 128) / 127 * FP8_MAX
    normalized = scaled / FP8_MAX  # [-1, 1]
    quantized = ((normalized * 127 + 128).round()).astype(np.uint8)

    # Clip to valid uint8 range
    quantized = np.clip(quantized, 0, 255)

    return quantized, scale


def compute_dequantize_fp8(
    tensor: np.ndarray,
    scale: float,
    dtype: "np.dtype[Any]" = np.dtype(np.float32),
) -> np.ndarray:
    """Dequantize FP8 tensor back to float.

    Args:
        tensor: FP8 tensor stored as uint8
        scale: Scale factor used during quantization
        dtype: Output data type (default: float32)

    Returns:
        Dequantized tensor in specified dtype

    Example:
        >>> original = np.random.randn(1024, 1024).astype(np.float32) * 10
        >>> quantized, scale = compute_quantize_fp8(original)
        >>> recovered = compute_dequantize_fp8(quantized, scale)
        >>> recovered.dtype
        dtype('float32')
    """
    # Convert uint8 back to normalized float [-1, 1]
    normalized = (tensor.astype(np.float32) - 128) / 127.0

    # Scale back to original range
    dequantized = normalized * FP8_MAX / scale

    return dequantized.astype(dtype)


def compute_fp8_scale(tensor: np.ndarray) -> float:
    """Compute optimal scale for FP8 quantization.

    The scale maps the maximum absolute value to FP8_MAX.

    Args:
        tensor: Input tensor

    Returns:
        Scale factor as float
    """
    max_abs = np.abs(tensor).max()
    if max_abs == 0:
        return 1.0
    return FP8_MAX / max_abs


def compute_fp8_quantization_error(
    original: np.ndarray,
    quantized: np.ndarray,
    scale: float,
) -> dict:
    """Compute quantization error metrics.

    Args:
        original: Original float tensor
        quantized: Quantized tensor (uint8)
        scale: Scale factor

    Returns:
        Dictionary with error metrics:
        - max_error: Maximum absolute error
        - mean_error: Mean absolute error
        - relative_error: Mean relative error
    """
    recovered = compute_dequantize_fp8(quantized, scale)

    abs_error = np.abs(original - recovered)
    max_error = float(abs_error.max())
    mean_error = float(abs_error.mean())

    # Avoid division by zero for relative error
    mask = np.abs(original) > 1e-6
    if mask.any():
        relative_error = float((abs_error[mask] / np.abs(original[mask])).mean())
    else:
        relative_error = 0.0

    return {
        "max_error": max_error,
        "mean_error": mean_error,
        "relative_error": relative_error,
    }
