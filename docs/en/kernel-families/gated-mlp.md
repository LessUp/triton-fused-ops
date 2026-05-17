---
title: Fused Gated MLP
description: The gated feed-forward family for SwiGLU and GeGLU style blocks
---

# Fused Gated MLP

`fused_gated_mlp` targets the gated feed-forward expansion stage used in modern Transformer blocks. It pairs two projections with an activation and an elementwise gate in one launch.

## Family summary

The code supports both **SwiGLU** (`silu`) and **GeGLU** (`gelu`) style activation choices. The user-facing APIs are:

- `fused_gated_mlp`
- `FusedGatedMLP`
- `reference_gated_mlp`

## Data contract

The family takes:

- `x` with shape `[batch, seq_len, hidden_dim]`,
- `gate_weight` with shape `[intermediate_dim, hidden_dim]`,
- `up_weight` with shape `[intermediate_dim, hidden_dim]`,
- an `activation` string selecting `silu` or `gelu`.

The important design variable is `intermediate_dim`, because it sets the expansion width and therefore the arithmetic and memory footprint of the family.

## What is fused

The implementation computes two matrix products from the same input tile, applies the selected activation, and multiplies the results to produce the gated activation output.

That is narrower than a full feed-forward network. Down-projection, residual handling, and block-level orchestration still live outside this Kernel family.

## Review questions

When reviewing this family, focus on:

1. whether the activation choice matches the target model,
2. whether the `intermediate_dim` is realistic for that model,
3. whether correctness checks compare against the reference implementation,
4. whether latency is interpreted in the context of the whole FFN path instead of just the fused sub-step.

## Benchmarking posture

Benchmarking here is less about a single theoretical bottleneck and more about the practical cost of repeated launches and intermediate movement. Use activation-specific experiments and compare against an unfused baseline that actually mirrors the model code you plan to replace.
