---
title: FP8 Quantization
description: "FP8 storage, scaling, and dequantization helpers"
---

# FP8 Quantization

The repository implements a compatibility-oriented FP8 path built around `uint8` storage plus explicit scale tensors.

## Format model

The code documents the path as FP8 E4M3-inspired quantization with:

- `FP8_MAX = 448.0`
- one scalar scale factor per tensor in the built-in runtime path
- `uint8` storage with an offset encoding around the signed range

Important scope note:

- The validation layer is aware of native float8 dtypes when PyTorch exposes them.
- The maintained kernel path in this repository still quantizes and interprets data through the `uint8` compatibility representation.

## `quantize_fp8`

```python
quantize_fp8(
    tensor: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

Requirements:

- `tensor` must be a contiguous CUDA tensor.
- Supported input dtypes: `float16`, `bfloat16`, `float32`.
- If `scale` is passed, it must be a positive CUDA scalar with dtype `float32`.

Returns:

- quantized tensor with dtype `torch.uint8`
- scale tensor with dtype `torch.float32`

Automatic scale rule:

```text
scale = 448.0 / max(abs(tensor))
```

If the tensor is all zeros, the implementation returns a scale of `1.0`.

## `dequantize_fp8`

```python
dequantize_fp8(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor
```

This reconstructs a floating-point tensor from the repository's FP8 storage format.

Typical use:

```python
import torch
from triton_ops import quantize_fp8, dequantize_fp8

x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
q, scale = quantize_fp8(x)
restored = dequantize_fp8(q, scale)
```

## Overflow-handling helper

The overflow-aware helper is implemented in the kernel module and is not re-exported from `triton_ops.__init__`.

Import path:

```python
from triton_ops.kernels.fp8_quantize import quantize_fp8_with_overflow_handling
```

Signature:

```python
quantize_fp8_with_overflow_handling(
    tensor: torch.Tensor,
    scale: torch.Tensor | None = None,
    max_attempts: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]
```

Behavior:

- Repeatedly checks whether `tensor * scale` fits the FP8 range.
- Halves the scale when it does not fit.
- Raises `NumericalOverflowError` if the retries still fail.

## `FP8Format`

`FP8Format` lives in `triton_ops.models` and is also exported from the root package.

Useful members:

- `FP8Format.max_value`
- `FP8Format.min_normal`
- `FP8Format.compute_scale(tensor)`
- `FP8Format.compute_scale_per_channel(tensor, dim=0)`
- `FP8Format.is_in_range(tensor, scale)`

Important practical note:

- The built-in `fp8_gemm` path consumes scalar scales, not per-channel scales.
- `compute_scale_per_channel` is still useful when you build custom quantization workflows outside the current GEMM interface.

## Recommended usage patterns

- Use `quantize_fp8` + `fp8_gemm` for explicit control of storage and scales.
- Use `fp8_gemm(a, b)` directly when you want the simplest path and are comfortable with automatic quantization.
- Keep numerically sensitive normalization steps in higher precision.
- Validate accuracy against an FP16 baseline for your own workload before promoting FP8 broadly.
