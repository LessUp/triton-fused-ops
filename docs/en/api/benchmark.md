---
layout: default
title: "Benchmark API — Triton Fused Ops"
description: "API reference for benchmark tools - performance measurement and correctness verification"
---

# Benchmark API Reference

This document provides API reference for the benchmark suite.

---

## Overview

The benchmark suite provides tools for:
- Performance measurement with synchronization
- Correctness verification against PyTorch reference
- Report generation
- Benchmark orchestration

---

## Benchmark Functions

### benchmark_kernel

Benchmark a kernel function with proper warmup and synchronization.

```python
from triton_ops.benchmark.suite import benchmark_kernel

results = benchmark_kernel(
    kernel_fn=my_kernel,
    args=(input_tensor, weight),
    warmup=10,
    iterations=100,
    device='cuda',
)

print(f"Mean latency: {results['mean_ms']:.3f} ms")
print(f"Std dev: {results['std_ms']:.3f} ms")
```

### compare_correctness

Compare kernel output against PyTorch reference.

```python
from triton_ops.benchmark.correctness import compare_correctness

is_correct, max_error = compare_correctness(
    kernel_fn=fused_rmsnorm_rope,
    reference_fn=pytorch_rmsnorm_rope,
    args=(x, weight, cos, sin),
    rtol=1e-3,
    atol=1e-5,
)

if is_correct:
    print(f"✅ Correct! Max error: {max_error:.6f}")
else:
    print(f"❌ Incorrect! Max error: {max_error:.6f}")
```

---

## Benchmark Report

Generate formatted benchmark reports.

```python
from triton_ops.benchmark.report import BenchmarkReport

report = BenchmarkReport()

# Add results
report.add_result(
    name="RMSNorm+RoPE",
    config={"batch": 8, "seq_len": 2048},
    metrics={"latency_ms": 0.89, "speedup": 3.2},
)

# Generate report
print(report.to_markdown())
print(report.to_json())
```

---

## Performance Metrics

### Latency Measurement

```python
import torch
import time

def measure_latency(fn, *args, warmup=10, iterations=100):
    """Measure kernel latency with proper synchronization."""
    
    # Warmup
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iterations):
        start.record()
        output = fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
    }
```

### Memory Bandwidth Calculation

```python
def compute_bandwidth(tensor_size_bytes, latency_ms):
    """Compute memory bandwidth in GB/s."""
    seconds = latency_ms / 1000
    gigabytes = tensor_size_bytes / (1024**3)
    return gigabytes / seconds
```

---

<div align="center">

**[⬆ Back to Top](#benchmark-api-reference)** | **[← Back to API Index](./)**

</div>
