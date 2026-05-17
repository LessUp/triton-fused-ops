---
title: Kernel Families
description: User-facing fused operation families and how to compare them
---

# Kernel Families

In this repository, the Kernel family is the unit that matters most to users. Each family is a deployable story with its own contract surface, proof strategy, and performance interpretation.

## Comparison matrix

| Kernel family | Main APIs | Workload story | Evidence surface |
| :-- | :-- | :-- | :-- |
| [Fused RMSNorm + RoPE](/en/kernel-families/rmsnorm-rope) | `fused_rmsnorm_rope`, `FusedRMSNormRoPE` | Remove an intermediate normalization write before rotary embedding | Validation + reference implementation + benchmark comparison |
| [Fused Gated MLP](/en/kernel-families/gated-mlp) | `fused_gated_mlp`, `FusedGatedMLP` | Fuse paired projections, activation, and gating into one launch | Validation + reference implementation + activation-aware measurement |
| [FP8 GEMM](/en/kernel-families/fp8-stack) | `fp8_gemm`, `FP8Linear` | Trade precision format and explicit scale management for storage and throughput goals | Quantization helpers + scale handling + GEMM benchmarking |
| FP8 quantization utilities | `quantize_fp8`, `dequantize_fp8` | Make the FP8 path inspectable rather than magical | Overflow handling, scale validation, round-trip semantics |

## How to read these pages

For each family, ask four questions:

1. What bottleneck is the family trying to move?
2. What tensors and contracts must already exist?
3. What reference path checks the semantics?
4. Which Benchmarking and Performance metrics are the right ones to use?

If a page cannot answer those four questions, it is not yet doing its job.
