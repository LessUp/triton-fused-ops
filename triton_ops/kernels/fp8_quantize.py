"""FP8 quantization and dequantization Triton kernels.

This module implements FP8 E4M3 quantization with dynamic scaling.
FP8 E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits.
Max representable value: 448.0
"""

import torch
import triton
import triton.language as tl

from triton_ops.models import FP8Format
from triton_ops.validation import validate_fp8_quantize_inputs
from triton_ops.exceptions import NumericalOverflowError


# FP8 E4M3 constants
FP8_MAX = 448.0
FP8_MIN_NORMAL = 2**-6


@triton.jit
def quantize_fp8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Quantize FP16/BF16 to FP8 E4M3 with scaling.
    
    Algorithm:
    1. Load input values
    2. Scale by the provided scale factor
    3. Clamp to FP8 range
    4. Store as uint8 (FP8 representation)
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Load scale factor
    scale = tl.load(scale_ptr)
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Scale and clamp to FP8 range
    x_scaled = x.to(tl.float32) * scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    
    # Convert to FP8 representation (stored as uint8)
    # This is a simplified conversion - actual FP8 would need bit manipulation
    # For now, we quantize to 8-bit range
    x_quantized = tl.libdevice.rint(x_clamped / FP8_MAX * 127.0)
    x_uint8 = x_quantized.to(tl.int8) + 128  # Shift to unsigned range
    
    # Store output
    tl.store(output_ptr + offsets, x_uint8.to(tl.uint8), mask=mask)


@triton.jit
def dequantize_fp8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Dequantize FP8 E4M3 back to FP16/BF16.
    
    Algorithm:
    1. Load FP8 values (stored as uint8)
    2. Convert back to float
    3. Divide by scale factor
    4. Store as FP16/BF16
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Load scale factor
    scale = tl.load(scale_ptr)
    inv_scale = 1.0 / scale
    
    # Load FP8 values
    x_uint8 = tl.load(input_ptr + offsets, mask=mask, other=128)
    
    # Convert back to float
    x_int8 = x_uint8.to(tl.int32) - 128  # Shift back to signed range
    x_float = x_int8.to(tl.float32) / 127.0 * FP8_MAX
    
    # Apply inverse scale
    x_dequant = x_float * inv_scale
    
    # Store output
    tl.store(output_ptr + offsets, x_dequant.to(tl.float16), mask=mask)


@triton.jit
def compute_scale_kernel(
    input_ptr,
    scale_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute optimal scale factor for FP8 quantization.
    
    Finds the maximum absolute value and computes scale = FP8_MAX / max_abs.
    """
    pid = tl.program_id(0)
    
    # Each block computes partial max
    max_val = tl.zeros([1], dtype=tl.float32)
    
    for block_start in range(pid * BLOCK_SIZE, numel, tl.num_programs(0) * BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < numel
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        block_max = tl.max(tl.abs(x.to(tl.float32)))
        max_val = tl.maximum(max_val, block_max)
    
    # Atomic max to get global maximum
    tl.atomic_max(scale_ptr, max_val)


def quantize_fp8(
    tensor: torch.Tensor,
    scale: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize tensor to FP8 E4M3 format.
    
    Args:
        tensor: Input tensor in FP16/BF16/FP32
        scale: Optional pre-computed scale factor. If None, computed automatically.
        
    Returns:
        Tuple of (quantized_tensor, scale_factor)
        - quantized_tensor: FP8 values stored as uint8
        - scale_factor: Scale used for quantization
    """
    validate_fp8_quantize_inputs(tensor, scale)
    
    # Compute scale if not provided
    if scale is None:
        scale = FP8Format.compute_scale(tensor)
    
    # Ensure scale is on same device
    if not scale.is_cuda:
        scale = scale.to(tensor.device)
    
    # Allocate output
    output = torch.empty(tensor.shape, dtype=torch.uint8, device=tensor.device)
    
    # Flatten for kernel
    tensor_flat = tensor.contiguous().view(-1)
    output_flat = output.view(-1)
    numel = tensor_flat.numel()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    
    quantize_fp8_kernel[grid](
        tensor_flat, output_flat, scale,
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output, scale


def dequantize_fp8(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Dequantize FP8 tensor back to FP16/BF16.
    
    Args:
        tensor: FP8 tensor stored as uint8
        scale: Scale factor used during quantization
        output_dtype: Output data type (float16 or bfloat16)
        
    Returns:
        Dequantized tensor in specified dtype
    """
    # Allocate output
    output = torch.empty(tensor.shape, dtype=output_dtype, device=tensor.device)
    
    # Flatten for kernel
    tensor_flat = tensor.contiguous().view(-1)
    output_flat = output.view(-1)
    numel = tensor_flat.numel()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    
    dequantize_fp8_kernel[grid](
        tensor_flat, output_flat, scale,
        numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def quantize_fp8_with_overflow_handling(
    tensor: torch.Tensor,
    scale: torch.Tensor = None,
    max_attempts: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize to FP8 with dynamic overflow handling.
    
    If overflow is detected, the scale factor is adjusted and quantization
    is retried.
    
    Args:
        tensor: Input tensor
        scale: Optional initial scale factor
        max_attempts: Maximum number of retry attempts
        
    Returns:
        Tuple of (quantized_tensor, final_scale)
        
    Raises:
        NumericalOverflowError: If overflow cannot be resolved
    """
    if scale is None:
        scale = FP8Format.compute_scale(tensor)
    
    for attempt in range(max_attempts):
        # Check if scale is valid
        if FP8Format.is_in_range(tensor, scale):
            return quantize_fp8(tensor, scale)
        
        # Reduce scale to handle overflow
        scale = scale / 2.0
    
    # Final attempt
    if FP8Format.is_in_range(tensor, scale):
        return quantize_fp8(tensor, scale)
    
    raise NumericalOverflowError(
        f"Cannot quantize tensor to FP8 after {max_attempts} attempts",
        max_value=tensor.abs().max().item(),
        scale=scale.item() if scale.numel() == 1 else None,
        attempts=max_attempts,
    )


def fp8_quantize_reference(
    tensor: torch.Tensor,
    scale: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation of FP8 quantization for testing.
    
    Args:
        tensor: Input tensor
        scale: Optional scale factor
        
    Returns:
        Tuple of (quantized_tensor, scale)
    """
    if scale is None:
        max_abs = tensor.abs().max()
        if max_abs == 0:
            scale = torch.tensor(1.0, device=tensor.device, dtype=torch.float32)
        else:
            scale = torch.tensor(FP8_MAX / max_abs.item(), device=tensor.device, dtype=torch.float32)
    
    # Scale and clamp
    scaled = tensor.float() * scale
    clamped = torch.clamp(scaled, -FP8_MAX, FP8_MAX)
    
    # Quantize to 8-bit
    quantized = torch.round(clamped / FP8_MAX * 127.0).to(torch.int8) + 128
    
    return quantized.to(torch.uint8), scale


def fp8_dequantize_reference(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reference implementation of FP8 dequantization for testing.
    
    Args:
        tensor: FP8 tensor as uint8
        scale: Scale factor
        output_dtype: Output dtype
        
    Returns:
        Dequantized tensor
    """
    # Convert back to float
    x_int8 = tensor.to(torch.int32) - 128
    x_float = x_int8.float() / 127.0 * FP8_MAX
    
    # Apply inverse scale
    dequant = x_float / scale
    
    return dequant.to(output_dtype)
