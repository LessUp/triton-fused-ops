---
layout: default
title: Performance Tuning
parent: Guides
grand_parent: Documentation
nav_order: 2
description: "How to measure, interpret, and tune performance in this repository"
---

# Performance Tuning

This page explains how to measure the shipped kernels correctly and how to reason about tuning work around them.

## Start with the right question

The repository contains three different performance stories:

- `fused_rmsnorm_rope`: primarily a memory-traffic reduction story,
- `fused_gated_mlp`: a fusion and launch-overhead reduction story,
- `fp8_gemm`: a quantization plus matrix-multiplication throughput story.

Treat them differently when benchmarking.

## Correct timing pattern

```python
import time
import torch
from triton_ops import fused_rmsnorm_rope

x = torch.randn(8, 2048, 4096, device="cuda", dtype=torch.float16)
weight = torch.ones(4096, device="cuda", dtype=torch.float16)
cos = torch.randn(2048, 64, device="cuda", dtype=torch.float16)
sin = torch.randn(2048, 64, device="cuda", dtype=torch.float16)

for _ in range(10):
    _ = fused_rmsnorm_rope(x, weight, cos, sin)
torch.cuda.synchronize()

start = time.perf_counter()
for _ in range(100):
    _ = fused_rmsnorm_rope(x, weight, cos, sin)
torch.cuda.synchronize()
end = time.perf_counter()

print((end - start) / 100 * 1000)
```

Always include:

- warmup runs,
- explicit synchronization before and after timing,
- representative shapes from your target model.

## Use the built-in benchmark layer when possible

`BenchmarkSuite` already wraps warmup, repeated execution, correctness verification, and report generation.

Use it when you want comparable outputs across multiple experiments.

## Interpreting metrics

The repository provides two metric helpers:

- `compute_gemm_metrics`
- `compute_elementwise_metrics`

Use them to distinguish between:

- computational throughput for GEMM-like work,
- effective bandwidth for elementwise or reduction-heavy kernels.

## Tuning custom kernels

`TritonAutoTuner` is useful when you own a custom kernel wrapper and want to search configuration space over:

- block sizes,
- warp counts,
- other keyword-parameterized launch choices.

The shipped kernel entry points do not automatically search config space during normal calls.

## Practical bottleneck checklist

### For `fused_rmsnorm_rope`

- check that RoPE cache shapes are correct and contiguous,
- keep hidden dimensions aligned with the head layout,
- treat memory traffic as the primary optimization target.

### For `fused_gated_mlp`

- benchmark on realistic intermediate dimensions,
- evaluate activation choice explicitly,
- remember that a full FFN includes additional surrounding work not measured by this kernel alone.

### For `fp8_gemm`

- compare auto-quantization against explicit pre-quantization,
- validate the numerical error against an FP16 baseline,
- benchmark representative matrix aspect ratios rather than only square matrices.

## What not to trust blindly

- A single GPU architecture result does not generalize to every deployment target.
- A benchmark without synchronization is not meaningful.
- A latency improvement on isolated kernels does not automatically translate into identical end-to-end model speedup.
