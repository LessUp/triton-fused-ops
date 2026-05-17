---
title: Runtime Contracts
description: Validation rules, typed errors, and launch-time expectations for Triton Fused Ops
---

# Runtime Contracts

Every shipped Kernel family assumes its inputs have already crossed a strict validation boundary. That boundary is the reason the launchers stay readable instead of becoming a mix of fast-path logic and scattered guard code.

## What the validators enforce

`triton_ops.validation` checks the things that would otherwise become silent footguns:

- tensors must be on the right device,
- supported dtypes must be used,
- shapes must agree,
- contiguous memory layouts must match expectations,
- scalar arguments such as epsilon or activation choices must be sane.

## Error vocabulary

| Error type | When it appears |
| :-- | :-- |
| `DeviceError` | A tensor is not on CUDA or is on the wrong device relative to its peers |
| `ShapeMismatchError` | Tensor shapes or derived dimensions disagree |
| `UnsupportedDtypeError` | The dtype is outside the family’s supported set |
| `NumericalOverflowError` | FP8 scaling or quantization cannot safely proceed |
| `TuningFailedError` | Auto-Tuning cannot find a valid configuration |

## Contract examples by family

### RMSNorm + RoPE

The validator checks device, dtype, contiguity, the shape of `weight`, agreement between `cos` and `sin`, and the relationship between `hidden_dim`, `head_dim`, and optional head count.

### Gated MLP

The validator checks paired projection shapes, supported activation selection, and the relation between `hidden_dim` and `intermediate_dim`.

### FP8 stack

The FP8 path validates shape compatibility, scale availability, supported storage dtypes, and output dtype choices.

## Why this matters for integration

When an integration fails, the goal is to fail early and with typed information. That is why the docs keep pointing back to runtime contracts: they are the stable explanation for what a Kernel family expects from upstream model code.
