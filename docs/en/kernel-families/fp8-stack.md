---
title: FP8 Stack
description: FP8 GEMM, FP8 quantization utilities, and scale-management strategy
---

# FP8 Stack

The FP8 path in this repository is a stack, not a single kernel. It includes FP8 quantization utilities, dequantization, `FP8 GEMM`, and the `FP8Linear` wrapper that caches quantized weights for repeated inference.

## Components in the stack

| Component | Role |
| :-- | :-- |
| `quantize_fp8` | Convert floating tensors to the repository’s FP8-compatible `uint8` storage format |
| `dequantize_fp8` | Recover FP16 or BF16 tensors from stored FP8 values and a scale |
| `fp8_gemm` | Multiply FP8 matrices with explicit scales and FP32 accumulation |
| `FP8Linear` | Wrap FP8 GEMM in a familiar linear-module interface |

## Format and storage model

The implementation is explicit about two things:

1. the modeled format is E4M3-oriented FP8 with repository constants anchored to `FP8Format`,
2. the practical storage path uses `uint8` values plus a scale tensor so the workflow is portable across environments that do not expose native FP8 tensors in the same way.

That explicitness is why this family is evidence-backed. You can read the scale handling, the overflow checks, and the conversion path directly.

## Scale management

`quantize_fp8` either accepts a provided scale or computes one through `FP8Format.compute_scale`. `dequantize_fp8` and `fp8_gemm` both depend on the same scale semantics. If the scale is invalid, the code raises numerical or validation errors instead of guessing.

## FP8 GEMM runtime story

`fp8_gemm` will auto-quantize floating inputs when needed, validate shape and dtype compatibility, and then launch a Triton GEMM that accumulates in FP32 before converting to the requested output dtype.

The kernel also uses grouped tile ordering to improve cache behavior. That is a systems-level optimization layered on top of the format and scale story.

## When to benchmark it

Use FP8 Benchmarking when the deployment question is really about memory footprint and matrix multiplication throughput, not when the dominant cost is somewhere else in the model. Bring Performance metrics only after the matrix shapes are explicit.
