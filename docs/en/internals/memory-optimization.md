---
layout: default
title: Memory Optimization
parent: Internals
grand_parent: Documentation
nav_order: 3
description: "How fusion and FP8 reduce memory pressure in the repository kernels"
---

# Memory Optimization

Much of the value in this repository comes from reducing unnecessary memory movement.

## Why fusion helps

Many transformer subgraphs are not limited only by arithmetic throughput. They also spend time moving intermediate tensors through HBM.

Fusion helps by keeping intermediate values closer to the executing kernel logic rather than writing them out and reading them back.

## `fused_rmsnorm_rope`

The main win is that normalized values do not have to be materialized as a separate global tensor before applying RoPE.

Practical effect:

- fewer HBM reads and writes,
- less launch overhead than two separate operations,
- a more bandwidth-oriented optimization profile.

## `fused_gated_mlp`

The input tile is reused to feed two projections in one kernel path, and the activation is applied before the final write.

This reduces the amount of intermediate state that would otherwise move through memory across separate operators.

## FP8 path

`fp8_gemm` also cuts memory pressure by using one-byte stored values for quantized matrices.

That affects both:

- storage footprint,
- bytes moved per matrix load.

The trade-off is controlled quantization error, which is why the FP8 guides emphasize baseline comparison.

## What to watch in practice

- tensor contiguity,
- shape choices that align well with the kernel's tiling strategy,
- whether the workload is truly dominated by the fused region rather than by surrounding model code.

## Bottom line

The repository is most effective when the removed memory traffic actually matters for the target workload. The biggest gains come from hot paths where intermediate tensors would otherwise be written to and read from HBM repeatedly.
