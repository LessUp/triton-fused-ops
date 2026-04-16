---
layout: default
title: "FP8 Quantization API — Triton Fused Ops"
description: "API reference for FP8 quantization utilities - quantize_fp8, dequantize_fp8, FP8Format"
---

# FP8 Quantization API Reference

This document provides detailed API reference for FP8 quantization utilities.

---

## Table of Contents

- [Overview](#overview)
- [quantize_fp8](#quantize_fp8)
- [dequantize_fp8](#dequantize_fp8)
- [quantize_fp8_with_overflow_handling](#quantize_fp8_with_overflow_handling)
- [FP8Format](#fp8format)

---

## Overview

FP8 (8-bit floating point) quantization reduces memory usage by 50% compared to FP16 while maintaining accuracy for inference. This library uses the **E4M3** format:

| Property | Value |
|----------|-------|
| Sign bits | 1 |
| Exponent bits | 4 |
| Mantissa bits | 3 |
| Max value | 448.0 |
| Min normal | 2^-6 ≈ 0.015625 |

### Storage Format

FP8 values are stored as `uint8` tensors for compatibility:

```python
# FP8 is stored as uint8
quantized: torch.uint8  # shape [...]
scale: torch.float32    # shape [1]
```

---

## quantize_fp8

Quantize tensor to FP8 E4M3 format.

### Syntax

```python
triton_ops.quantize_fp8(
    tensor: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor` | `torch.Tensor` | Input tensor in `float16`, `bfloat16`, or `float32`. Must be on CUDA. |
| `scale` | `Optional[torch.Tensor]` | Pre-computed scale factor. If `None`, computed automatically. |

### Returns

`tuple[torch.Tensor, torch.Tensor]`:
- **quantized**: FP8 values stored as `uint8`, same shape as input.
- **scale**: Scale factor used for quantization, shape `[1]`, dtype `float32`.

### Scale Computation

If `scale` is not provided, it is computed as:

```
scale = FP8_MAX / max(abs(tensor))
      = 448.0 / max(abs(tensor))
```

### Example

```python
import torch
from triton_ops import quantize_fp8

# Create input tensor
tensor = torch.randn(1024, 4096, device='cuda', dtype=torch.float16)
print(f"Original: {tensor.dtype}, {tensor.element_size()} bytes")

# Quantize to FP8
quantized, scale = quantize_fp8(tensor)
print(f"Quantized: {quantized.dtype}, {quantized.element_size()} bytes")
print(f"Scale: {scale.item():.6f}")

# Memory saved: 50%
# FP16: 2 bytes per element
# FP8:  1 byte per element
```

---

## dequantize_fp8

Dequantize FP8 tensor back to floating point.

### Syntax

```python
triton_ops.dequantize_fp8(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor` | `torch.Tensor` | FP8 tensor stored as `uint8`. |
| `scale` | `torch.Tensor` | Scale factor from quantization. |
| `output_dtype` | `torch.dtype` | Output dtype. Default: `torch.float16`. |

### Returns

`torch.Tensor` — Dequantized tensor in `output_dtype`.

### Example

```python
import torch
from triton_ops import quantize_fp8, dequantize_fp8

# Quantize
original = torch.randn(512, 512, device='cuda', dtype=torch.float16)
quantized, scale = quantize_fp8(original)

# Dequantize
recovered = dequantize_fp8(quantized, scale, output_dtype=torch.float16)

# Check reconstruction error
error = torch.abs(original - recovered).mean().item()
print(f"Mean reconstruction error: {error:.6f}")
```

---

## quantize_fp8_with_overflow_handling

Quantize to FP8 with dynamic overflow handling.

If overflow is detected (values exceed FP8 range after scaling), the scale factor is automatically adjusted and quantization is retried.

### Syntax

```python
triton_ops.quantize_fp8_with_overflow_handling(
    tensor: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    max_attempts: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor` | `torch.Tensor` | Input tensor. |
| `scale` | `Optional[torch.Tensor]` | Initial scale factor. If `None`, computed automatically. |
| `max_attempts` | `int` | Maximum retry attempts. Default: `3`. |

### Returns

`tuple[torch.Tensor, torch.Tensor]`:
- **quantized**: FP8 values.
- **final_scale**: Adjusted scale factor.

### Raises

| Exception | Condition |
|-----------|-----------|
| `NumericalOverflowError` | Cannot resolve overflow after `max_attempts`. |

### Example

```python
import torch
from triton_ops import quantize_fp8_with_overflow_handling

# Tensor with potential overflow
tensor = torch.randn(1024, 4096, device='cuda', dtype=torch.float16) * 500

# Safe quantization with overflow handling
try:
    quantized, scale = quantize_fp8_with_overflow_handling(tensor, max_attempts=3)
    print(f"Quantized successfully with scale: {scale.item():.6f}")
except NumericalOverflowError as e:
    print(f"Failed to quantize: {e}")
```

---

## FP8Format

Dataclass for FP8 E4M3 format specification and utilities.

### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `exponent_bits` | `int` | `4` | Number of exponent bits. |
| `mantissa_bits` | `int` | `3` | Number of mantissa bits. |
| `max_value` | `float` | `448.0` | Maximum representable value. |
| `min_normal` | `float` | `2**-6` | Smallest normal number. |

### Static Methods

#### compute_scale

```python
FP8Format.compute_scale(tensor: torch.Tensor) -> torch.Tensor
```

Compute optimal scaling factor for FP8 conversion.

**Returns:** `torch.Tensor` — Scale factor, shape `[1]`, dtype `float32`.

#### compute_scale_per_channel

```python
FP8Format.compute_scale_per_channel(
    tensor: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor
```

Compute per-channel scaling factors for FP8 conversion.

**Returns:** `torch.Tensor` — Per-channel scales, shape matching input except reduced dimension.

#### is_in_range

```python
FP8Format.is_in_range(
    tensor: torch.Tensor,
    scale: torch.Tensor,
) -> bool
```

Check if scaled tensor is within FP8 representable range.

**Returns:** `bool` — `True` if all values are within `[-448.0, 448.0]`.

### Example

```python
from triton_ops import FP8Format
import torch

# Format properties
print(f"FP8 max value: {FP8Format.max_value}")  # 448.0
print(f"FP8 min normal: {FP8Format.min_normal}")  # 0.015625

# Compute scale
tensor = torch.randn(512, 512, device='cuda', dtype=torch.float16)
scale = FP8Format.compute_scale(tensor)
print(f"Scale: {scale.item():.6f}")

# Check if in range
if FP8Format.is_in_range(tensor, scale):
    print("Tensor is within FP8 range after scaling")
```

---

## Best Practices

### 1. Use Automatic Scaling

For most use cases, let the library compute the scale automatically:

```python
quantized, scale = quantize_fp8(tensor)  # Automatic scaling
```

### 2. Cache Scale Factors

For repeated operations, compute scale once and reuse:

```python
# Compute scale once
_, scale = quantize_fp8(weight)

# Reuse for multiple inputs
for x in inputs:
    x_quantized, _ = quantize_fp8(x, scale=scale)
```

### 3. Check Accuracy

Always verify reconstruction error for your use case:

```python
quantized, scale = quantize_fp8(tensor)
recovered = dequantize_fp8(quantized, scale)
error = torch.abs(tensor - recovered).max().item()

if error > 1.0:
    print(f"Warning: Large quantization error: {error}")
```

### 4. Handle Edge Cases

Use overflow handling for tensors with extreme values:

```python
from triton_ops import quantize_fp8_with_overflow_handling, NumericalOverflowError

try:
    quantized, scale = quantize_fp8_with_overflow_handling(tensor)
except NumericalOverflowError:
    # Fallback to FP16 or clip tensor
    pass
```

---

<div align="center">

**[⬆ Back to Top](#fp8-quantization-api-reference)** | **[← Back to API Index](./)**

</div>
