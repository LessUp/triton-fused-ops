"""Reference implementations for FP8 quantization.

This module provides unified reference implementations for FP8 E4M3
quantization and dequantization.

FP8 E4M3 format:
    - 1 sign bit, 4 exponent bits, 3 mantissa bits
    - Maximum representable value: 448.0
    - Minimum normal: 2^-6 ≈ 0.015625

Both CPU (NumPy) and GPU (PyTorch) backends are supported.

Example:
    >>> from triton_ops.reference import quantize_fp8, dequantize_fp8
    >>> import numpy as np
    >>>
    >>> # CPU testing
    >>> tensor = np.random.randn(1024, 1024).astype(np.float32) * 10
    >>> quantized, scale = quantize_fp8(tensor, backend='cpu')
    >>> recovered = dequantize_fp8(quantized, scale, backend='cpu')
"""

import numpy as np
import torch

from triton_ops.models import FP8Format
from triton_ops.reference.base import (
    Backend,
    ensure_numpy,
    ensure_torch,
    validate_backend,
)


def quantize_fp8(
    tensor: np.ndarray | torch.Tensor,
    scale: np.ndarray | torch.Tensor | float | None = None,
    *,
    backend: Backend = "cpu",
) -> tuple[np.ndarray, float] | tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to FP8 E4M3 format.

    Args:
        tensor: Input tensor in float32/float16/bfloat16
        scale: Optional pre-computed scale factor. If None, computed automatically.
        backend: 'cpu' for NumPy, 'cuda' for PyTorch (default: 'cpu')

    Returns:
        Tuple of (quantized_tensor, scale_factor)
        - quantized_tensor: FP8 values stored as uint8
        - scale_factor: Scale used for quantization

    Example:
        >>> import numpy as np
        >>> tensor = np.random.randn(1024, 1024).astype(np.float32) * 10
        >>> quantized, scale = quantize_fp8(tensor, backend='cpu')
        >>> quantized.dtype
        dtype('uint8')
        >>> scale > 0
        True
    """
    validate_backend(backend)

    if backend == "cpu":
        return _quantize_fp8_cpu(tensor, scale)
    else:
        return _quantize_fp8_cuda(tensor, scale)


def _quantize_fp8_cpu(
    tensor: np.ndarray | torch.Tensor,
    scale: np.ndarray | torch.Tensor | float | None,
) -> tuple[np.ndarray, float]:
    """NumPy implementation of FP8 quantization."""
    tensor = ensure_numpy(tensor)

    # Compute scale if not provided
    if scale is None:
        max_abs = np.abs(tensor).max()
        if max_abs == 0:
            scale_val = 1.0
        else:
            scale_val = FP8Format.max_value / max_abs
    elif isinstance(scale, (np.ndarray, torch.Tensor)):
        scale_val = float(ensure_numpy(scale).item())
    else:
        scale_val = float(scale)

    # Scale the tensor
    scaled = tensor * scale_val

    # Clip to FP8 range
    scaled = np.clip(scaled, -FP8Format.max_value, FP8Format.max_value)

    # Convert to FP8-like representation
    # Map [-FP8_MAX, FP8_MAX] to [0, 255]
    normalized = scaled / FP8Format.max_value  # [-1, 1]
    quantized = ((normalized * 127 + 128).round()).astype(np.uint8)

    # Clip to valid uint8 range
    quantized = np.clip(quantized, 0, 255)

    return quantized, scale_val


def _quantize_fp8_cuda(
    tensor: np.ndarray | torch.Tensor,
    scale: np.ndarray | torch.Tensor | float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch implementation of FP8 quantization."""
    tensor = ensure_torch(tensor, device="cuda")

    # Compute scale if not provided
    if scale is None:
        max_abs = tensor.abs().max()
        if max_abs == 0:
            scale_tensor = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        else:
            scale_tensor = torch.tensor(
                FP8Format.max_value / max_abs.item(),
                device="cuda",
                dtype=torch.float32,
            )
    elif isinstance(scale, torch.Tensor):
        scale_tensor = scale.to(device="cuda", dtype=torch.float32)
    elif isinstance(scale, np.ndarray):
        scale_tensor = torch.from_numpy(scale).to(device="cuda", dtype=torch.float32)
    else:
        scale_tensor = torch.tensor(scale, device="cuda", dtype=torch.float32)

    # Scale and clamp
    scaled = tensor.float() * scale_tensor
    clamped = torch.clamp(scaled, -FP8Format.max_value, FP8Format.max_value)

    # Quantize to 8-bit
    quantized = torch.round(clamped / FP8Format.max_value * 127).to(torch.int8) + 128

    return quantized.to(torch.uint8), scale_tensor


def dequantize_fp8(
    tensor: np.ndarray | torch.Tensor,
    scale: np.ndarray | torch.Tensor | float,
    output_dtype: np.dtype | torch.dtype = np.float32,
    *,
    backend: Backend = "cpu",
) -> np.ndarray | torch.Tensor:
    """Dequantize FP8 tensor back to float.

    Args:
        tensor: FP8 tensor stored as uint8
        scale: Scale factor used during quantization
        output_dtype: Output data type (default: float32)
        backend: 'cpu' for NumPy, 'cuda' for PyTorch (default: 'cpu')

    Returns:
        Dequantized tensor in specified dtype

    Example:
        >>> import numpy as np
        >>> original = np.random.randn(1024, 1024).astype(np.float32) * 10
        >>> quantized, scale = quantize_fp8(original, backend='cpu')
        >>> recovered = dequantize_fp8(quantized, scale, backend='cpu')
        >>> recovered.dtype
        dtype('float32')
    """
    validate_backend(backend)

    if backend == "cpu":
        return _dequantize_fp8_cpu(tensor, scale, output_dtype)
    else:
        return _dequantize_fp8_cuda(tensor, scale, output_dtype)


def _dequantize_fp8_cpu(
    tensor: np.ndarray | torch.Tensor,
    scale: np.ndarray | torch.Tensor | float,
    output_dtype: np.dtype | torch.dtype,
) -> np.ndarray:
    """NumPy implementation of FP8 dequantization."""
    tensor = ensure_numpy(tensor)

    # Get scale value
    if isinstance(scale, (np.ndarray, torch.Tensor)):
        scale_val = float(ensure_numpy(scale).item())
    else:
        scale_val = float(scale)

    # Convert uint8 back to normalized float [-1, 1]
    normalized = (tensor.astype(np.float32) - 128) / 127.0

    # Scale back to original range
    dequantized = normalized * FP8Format.max_value / scale_val

    # Convert dtype
    if isinstance(output_dtype, torch.dtype):
        dtype_map = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.bfloat16: np.float32,  # numpy doesn't have bfloat16
        }
        output_dtype = dtype_map.get(output_dtype, np.float32)

    return dequantized.astype(output_dtype)


def _dequantize_fp8_cuda(
    tensor: np.ndarray | torch.Tensor,
    scale: np.ndarray | torch.Tensor | float,
    output_dtype: np.dtype | torch.dtype,
) -> torch.Tensor:
    """PyTorch implementation of FP8 dequantization."""
    tensor = ensure_torch(tensor, device="cuda")

    # Get scale tensor
    if isinstance(scale, torch.Tensor):
        scale_tensor = scale.to(device="cuda", dtype=torch.float32)
    elif isinstance(scale, np.ndarray):
        scale_tensor = torch.from_numpy(scale).to(device="cuda", dtype=torch.float32)
    else:
        scale_tensor = torch.tensor(scale, device="cuda", dtype=torch.float32)

    # Convert back to float
    x_int8 = tensor.to(torch.int32) - 128
    x_float = x_int8.float() / 127.0 * FP8Format.max_value

    # Apply inverse scale
    dequant = x_float / scale_tensor

    # Convert dtype
    if isinstance(output_dtype, np.dtype):
        dtype_map = {
            np.dtype(np.float16): torch.float16,
            np.dtype(np.float32): torch.float32,
        }
        output_dtype = dtype_map.get(output_dtype, torch.float32)

    return dequant.to(dtype=output_dtype)


def compute_fp8_scale(
    tensor: np.ndarray | torch.Tensor,
    *,
    backend: Backend = "cpu",
) -> float | torch.Tensor:
    """Compute optimal scale for FP8 quantization.

    The scale maps the maximum absolute value to FP8_MAX.

    Args:
        tensor: Input tensor
        backend: 'cpu' for NumPy, 'cuda' for PyTorch (default: 'cpu')

    Returns:
        Scale factor (float for CPU, torch.Tensor for CUDA)
    """
    validate_backend(backend)

    if backend == "cpu":
        tensor = ensure_numpy(tensor)
        max_abs = np.abs(tensor).max()
        if max_abs == 0:
            return 1.0
        return float(FP8Format.max_value / max_abs)
    else:
        tensor = ensure_torch(tensor, device="cuda")
        max_abs = tensor.abs().max()
        if max_abs == 0:
            return torch.tensor(1.0, device="cuda", dtype=torch.float32)
        return torch.tensor(
            FP8Format.max_value / max_abs.item(),
            device="cuda",
            dtype=torch.float32,
        )


def fp8_gemm(
    a: np.ndarray | torch.Tensor,
    b: np.ndarray | torch.Tensor,
    a_scale: np.ndarray | torch.Tensor | float | None = None,
    b_scale: np.ndarray | torch.Tensor | float | None = None,
    output_dtype: np.dtype | torch.dtype = np.float32,
    *,
    backend: Backend = "cpu",
) -> np.ndarray | torch.Tensor:
    """Reference implementation of FP8 GEMM.

    Computes C = A @ B where A and B are FP8 quantized matrices.

    Args:
        a: First matrix (FP8 as uint8 or float)
        b: Second matrix (FP8 as uint8 or float)
        a_scale: Scale factor for A (required if A is uint8)
        b_scale: Scale factor for B (required if B is uint8)
        output_dtype: Output dtype
        backend: 'cpu' for NumPy, 'cuda' for PyTorch (default: 'cpu')

    Returns:
        Result matrix C

    Example:
        >>> import numpy as np
        >>> a = np.random.randn(128, 256).astype(np.float32) * 0.1
        >>> b = np.random.randn(256, 128).astype(np.float32) * 0.1
        >>> c = fp8_gemm(a, b, backend='cpu')
        >>> c.shape
        (128, 128)
    """
    validate_backend(backend)

    if backend == "cpu":
        return _fp8_gemm_cpu(a, b, a_scale, b_scale, output_dtype)
    else:
        return _fp8_gemm_cuda(a, b, a_scale, b_scale, output_dtype)


def _fp8_gemm_cpu(
    a: np.ndarray | torch.Tensor,
    b: np.ndarray | torch.Tensor,
    a_scale: np.ndarray | torch.Tensor | float | None,
    b_scale: np.ndarray | torch.Tensor | float | None,
    output_dtype: np.dtype | torch.dtype,
) -> np.ndarray:
    """NumPy implementation of FP8 GEMM."""
    a = ensure_numpy(a)
    b = ensure_numpy(b)

    # Handle FP8 inputs (uint8)
    if a.dtype == np.uint8:
        if a_scale is None:
            raise ValueError("a_scale is required when a is uint8")
        a_scale_val = (
            float(ensure_numpy(a_scale).item())
            if isinstance(a_scale, (np.ndarray, torch.Tensor))
            else float(a_scale)
        )
        a_int = a.astype(np.int32) - 128
        a_float = a_int / 127.0 * FP8Format.max_value / a_scale_val
    else:
        a_float = a.astype(np.float32)

    if b.dtype == np.uint8:
        if b_scale is None:
            raise ValueError("b_scale is required when b is uint8")
        b_scale_val = (
            float(ensure_numpy(b_scale).item())
            if isinstance(b_scale, (np.ndarray, torch.Tensor))
            else float(b_scale)
        )
        b_int = b.astype(np.int32) - 128
        b_float = b_int / 127.0 * FP8Format.max_value / b_scale_val
    else:
        b_float = b.astype(np.float32)

    # Compute matrix multiplication
    c = a_float @ b_float

    # Convert dtype
    if isinstance(output_dtype, torch.dtype):
        dtype_map = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.bfloat16: np.float32,
        }
        output_dtype = dtype_map.get(output_dtype, np.float32)

    return c.astype(output_dtype)


def _fp8_gemm_cuda(
    a: np.ndarray | torch.Tensor,
    b: np.ndarray | torch.Tensor,
    a_scale: np.ndarray | torch.Tensor | float | None,
    b_scale: np.ndarray | torch.Tensor | float | None,
    output_dtype: np.dtype | torch.dtype,
) -> torch.Tensor:
    """PyTorch implementation of FP8 GEMM."""
    a = ensure_torch(a, device="cuda")
    b = ensure_torch(b, device="cuda")

    # Handle FP8 inputs (uint8)
    if a.dtype == torch.uint8:
        if a_scale is None:
            raise ValueError("a_scale is required when a is uint8")
        if isinstance(a_scale, torch.Tensor):
            a_scale_tensor = a_scale.to(device="cuda", dtype=torch.float32)
        elif isinstance(a_scale, np.ndarray):
            a_scale_tensor = torch.from_numpy(a_scale).to(device="cuda", dtype=torch.float32)
        else:
            a_scale_tensor = torch.tensor(a_scale, device="cuda", dtype=torch.float32)
        a_int = a.to(torch.int32) - 128
        a_float = a_int.float() / 127.0 * FP8Format.max_value / a_scale_tensor
    else:
        a_float = a.float()

    if b.dtype == torch.uint8:
        if b_scale is None:
            raise ValueError("b_scale is required when b is uint8")
        if isinstance(b_scale, torch.Tensor):
            b_scale_tensor = b_scale.to(device="cuda", dtype=torch.float32)
        elif isinstance(b_scale, np.ndarray):
            b_scale_tensor = torch.from_numpy(b_scale).to(device="cuda", dtype=torch.float32)
        else:
            b_scale_tensor = torch.tensor(b_scale, device="cuda", dtype=torch.float32)
        b_int = b.to(torch.int32) - 128
        b_float = b_int.float() / 127.0 * FP8Format.max_value / b_scale_tensor
    else:
        b_float = b.float()

    # Compute matrix multiplication
    c = torch.matmul(a_float, b_float)

    # Convert dtype
    if isinstance(output_dtype, np.dtype):
        dtype_map = {
            np.dtype(np.float16): torch.float16,
            np.dtype(np.float32): torch.float32,
        }
        output_dtype = dtype_map.get(output_dtype, torch.float32)

    return c.to(dtype=output_dtype)
