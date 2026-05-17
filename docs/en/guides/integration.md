---
title: Integration Guide
description: How to integrate Triton Fused Ops into larger inference code without violating runtime contracts
---

# Integration Guide

This guide is about replacing a local hot path with a Kernel family while keeping runtime contracts explicit.

## Start from the boundary, not the kernel

Before touching model code, answer three questions:

1. Which tensors does the existing block already own?
2. Which runtime contracts can it already satisfy?
3. Is the replacement a full block swap or only a partial hot-path substitution?

That framing avoids a common mistake: choosing a kernel first and then forcing model code to produce missing state.

## Functional APIs vs module wrappers

| Option | Use when | Relevant APIs |
| :-- | :-- | :-- |
| Functional call | You already own weights, caches, and tensor orchestration | `fused_rmsnorm_rope`, `fused_gated_mlp`, `fp8_gemm`, `quantize_fp8`, `dequantize_fp8` |
| Module wrapper | You want a reusable `nn.Module` boundary | `FusedRMSNormRoPE`, `FusedGatedMLP`, `FP8Linear` |

## Runtime contracts you must respect

The launchers assume the upstream code delivers tensors on CUDA with supported dtypes, compatible shapes, and contiguous memory layouts. If that is not already true, fix the calling code before blaming the Kernel family.

## Family-by-family notes

### `FusedRMSNormRoPE`

`FusedRMSNormRoPE` belongs where the model already has RoPE caches or can produce them cheaply. It is not a generic normalization drop-in because the contract includes `cos` and `sin`.

### `FusedGatedMLP`

This wrapper covers the gated expansion stage, not the full feed-forward block. Keep the down-projection and residual path outside the module unless you are building a broader abstraction on top.

### `FP8Linear`

`FP8Linear` is an inference-oriented convenience wrapper around the FP8 stack. It works best when weights are stable enough that quantized storage and cached scales are a good trade.

## Integration checklist

- verify runtime contracts against `triton_ops.validation`,
- compare outputs with the reference path before rollout,
- benchmark the exact model boundary you changed,
- keep the rest of the model untouched until the local replacement is understood.
