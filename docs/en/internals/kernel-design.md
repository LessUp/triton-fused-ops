---
layout: default
title: Kernel Design
parent: Internals
grand_parent: Documentation
nav_order: 2
description: "Fusion patterns, tiling choices, and kernel-level implementation ideas"
---

# Kernel Design

This page explains the main implementation ideas used by the repository's Triton kernels.

## `fused_rmsnorm_rope`

The central idea is to keep the normalized values in registers long enough to apply RoPE before writing the final output.

Design goals:

- compute RMS statistics per row,
- apply the weight scale,
- immediately rotate the head pairs,
- write only the final tensor.

Why it matters:

- the unfused path would typically materialize an intermediate normalized tensor,
- the fused path avoids that extra global-memory traffic.

## `fused_gated_mlp`

The kernel computes two projections for the same input tile:

- gate projection,
- up projection.

It then applies the selected activation to the gate projection and multiplies it with the up projection result:

```text
output = activation(gate_proj(x)) * up_proj(x)
```

This combines projection and activation work in one launch instead of splitting them across separate operations.

## `fp8_gemm`

The GEMM kernel works with the repository's FP8 compatibility representation:

- values stored in `uint8`,
- explicit scales loaded from scalar tensors,
- FP32 accumulation,
- half-precision output path.

The code also uses grouped output-tile ordering to improve cache locality.

## Tiling heuristics

The current launchers choose block sizes heuristically from problem dimensions rather than from online autotuning during each call.

Examples:

- larger tiles for larger GEMMs,
- smaller `BLOCK_K` when the reduction dimension is smaller,
- simple fixed tile choices in the current fused Gated MLP path.

This keeps the runtime path predictable and small, while leaving more elaborate search to the generic autotuner tools.

## Reference implementations matter

Each kernel module also carries a reference implementation in plain PyTorch. Those references are important because they provide:

- correctness comparisons,
- a readable mathematical baseline,
- benchmark verification inputs.

The design philosophy is not just speed, but speed with a local verification path.
