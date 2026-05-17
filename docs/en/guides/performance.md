---
title: Performance Guide
description: How to benchmark, tune, and interpret Triton Fused Ops without mixing responsibilities
---

# Performance Guide

Performance work in this repository is easier once three terms stay separate: **Benchmarking**, **Auto-Tuning**, and **Performance metrics**.

## The three layers

| Concern | Primary tools | What it answers |
| :-- | :-- | :-- |
| Benchmarking | `BenchmarkSuite`, `CorrectnessVerifier`, `PerformanceReport` | How fast was a Kernel family under a declared method, and was it still correct? |
| Auto-Tuning | `TritonAutoTuner`, `ConfigCache`, preset config spaces | Which launch configuration minimizes latency? |
| Performance metrics | `PerformanceProfile`, `MetricsCalculator`, `compute_metrics` | Given shape context, what throughput or bandwidth does the observed latency imply? |

## Benchmarking first

Start with Benchmarking because it answers the most basic question: is the measured kernel both fast **and** correct?

A serious Benchmarking setup should include:

- warmup runs,
- explicit `torch.cuda.synchronize()` around timing,
- representative model shapes,
- correctness checks against the reference implementation.

`BenchmarkSuite` already packages that workflow, which is why it should be the default measurement tool when you want comparable evidence.

## Where Auto-Tuning fits

Auto-Tuning is for searching launch parameters such as block sizes, warp counts, or related configuration choices. It should not be confused with end-to-end benchmarking.

A useful rule is:

- use **Auto-Tuning** when you own the callable and want the best configuration,
- use **Benchmarking** when you need evidence attached to a reported result.

## Interpreting Performance metrics

Latency by itself is not the full story. `triton_ops.performance` computes Performance metrics only once the problem shape is known.

- elementwise-style work is usually read through effective bandwidth,
- GEMM-like work is usually read through throughput,
- zero-shape-context claims should stay latency-only.

## Family-specific cautions

### RMSNorm + RoPE

Treat this as a memory-sensitive family. Benchmarking should emphasize representative sequence lengths and hidden sizes.

### Gated MLP

Treat this as a launch-count and fusion-efficiency story. Include the activation choice in the experimental record.

### FP8 stack

Treat this as a format-and-scale story plus matrix multiplication behavior. Record whether inputs were pre-quantized or auto-quantized.

## What to avoid

- Do not present Auto-Tuning results as if they were complete Benchmarking evidence.
- Do not report Performance metrics without stating the shape assumptions.
- Do not extrapolate isolated kernel wins directly to whole-model speedup.
