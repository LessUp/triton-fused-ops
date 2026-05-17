---
title: Fused RMSNorm + RoPE
description: The fused normalization and rotary-embedding family in Triton Fused Ops
---

# Fused RMSNorm + RoPE

`fused_rmsnorm_rope` is the clearest fusion story in the repository: normalize once, rotate once, and avoid materializing an intermediate normalized tensor in HBM.

## What the family does

The implementation combines two operations that normally appear back-to-back in Transformer attention preparation:

1. **RMSNorm** using the learned weight vector,
2. **RoPE** using precomputed `cos` and `sin` embeddings.

The user-facing family is exposed as:

- `fused_rmsnorm_rope`
- `FusedRMSNormRoPE`
- `reference_fused_rmsnorm_rope` for correctness comparisons

## Why it is a distinct Kernel family

The family has one dominant claim: reduce memory traffic by keeping the normalized values close to the computation that immediately consumes them.

That is why the family should be evaluated primarily as a memory-traffic story rather than as a generic “more fusion is better” slogan.

## Contract surface

The runtime boundary is more specific than a normal norm layer:

| Input | Expected role |
| :-- | :-- |
| `x` | `[batch, seq_len, hidden_dim]` input activations |
| `weight` | `[hidden_dim]` RMSNorm weight |
| `cos`, `sin` | RoPE tables shaped for the sequence and head dimension |
| `num_heads` | Optional explicit override if you do not want it inferred |

The validators check CUDA placement, dtype support, contiguity, shape agreement, and divisibility relationships between `hidden_dim` and head geometry.

## Validation and evidence

The relevant evidence surfaces are straightforward:

- validation is centralized in `triton_ops.validation` rather than hidden inside the launch path,
- `validate_rmsnorm_rope_inputs` defines the runtime constraints,
- the reference implementation gives a readable mathematical baseline,
- Benchmarking should compare the fused path against a clear unfused baseline.

## Benchmarking guidance

Use Benchmarking to answer a narrow question: does the fusion remove enough traffic to matter for realistic sequence and hidden sizes?

That means performance reviews should record:

- warmup and synchronized timing,
- representative `seq_len` and `hidden_dim`,
- correctness checks against the reference path,
- Performance metrics that emphasize memory-sensitive behavior when appropriate.

## Integration notes

The `FusedRMSNormRoPE` module is easiest to integrate when the caller already owns RoPE caches. It is not a drop-in replacement for every normalization site in a model; it belongs specifically at the point where normalization and rotary embedding are adjacent in the attention pipeline.
