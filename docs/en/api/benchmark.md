---
title: Benchmarking
description: "Benchmark orchestration, correctness verification, and report generation"
---

# Benchmarking

The benchmark layer is organized around classes and helper functions, not standalone root-level benchmark functions.

## `BenchmarkSuite`

```python
BenchmarkSuite(
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    rtol: float = 1e-3,
    atol: float = 1e-5,
)
```

Main methods:

- `benchmark_kernel(...)`
- `compare_with_pytorch(...)`
- `benchmark_rmsnorm_rope(...)`
- `benchmark_gated_mlp(...)`
- `benchmark_fp8_gemm(...)`
- `generate_report(format="text" | "json")`
- `save_report(filepath, format="text" | "json")`

Example:

```python
import torch
from triton_ops import BenchmarkSuite, fused_rmsnorm_rope
from triton_ops.kernels.rmsnorm_rope import fused_rmsnorm_rope_reference

suite = BenchmarkSuite(warmup_runs=5, benchmark_runs=20)

x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
weight = torch.ones(4096, device="cuda", dtype=torch.float16)
cos = torch.randn(128, 64, device="cuda", dtype=torch.float16)
sin = torch.randn(128, 64, device="cuda", dtype=torch.float16)

result = suite.benchmark_kernel(
    fused_rmsnorm_rope,
    fused_rmsnorm_rope_reference,
    "fused_rmsnorm_rope",
    (2, 128, 4096),
    x,
    weight,
    cos,
    sin,
)
```

## `CorrectnessVerifier`

```python
CorrectnessVerifier(rtol: float = 1e-3, atol: float = 1e-5)
```

Useful methods:

- `verify(actual, expected) -> tuple[bool, dict]`
- `verify_allclose(actual, expected) -> bool`
- `compute_relative_error(actual, expected) -> float`

The detailed `verify` method returns aggregate statistics such as maximum absolute difference, mean relative difference, and element-count violations.

## Standalone correctness helpers

Available from `triton_ops.benchmark.correctness`:

- `verify_fp8_accuracy(fp8_result, fp16_baseline, max_relative_error=0.01)`
- `verify_nan_inf_propagation(output, input_has_nan, input_has_inf)`

These are useful when you want direct numerical checks outside `BenchmarkSuite`.

## Report objects

`triton_ops.benchmark.report` defines:

- `BenchmarkResult`
- `ComparisonResult`
- `PerformanceReport`

`PerformanceReport` can emit:

- human-readable text via `generate_text_report()`
- JSON via `generate_json_report()`

## Important accuracy note

This repository's benchmark utilities are GPU-oriented in the specialized benchmark methods because they allocate tensors on CUDA. For CPU-safe validation of repository health, use the test and build commands rather than the GPU benchmark suite.

## Related metric helpers

For throughput and bandwidth, use `PerformanceProfile`:

```python
from triton_ops import PerformanceProfile

# GEMM metrics
profile = PerformanceProfile.gemm(M=1024, N=4096, K=4096)
metrics = profile.metrics(latency_ms=0.5)

# Elementwise metrics
profile = PerformanceProfile.elementwise(numel=1024*4096)
metrics = profile.metrics(latency_ms=0.1)
```

See the [Auto-Tuning](/en/api/autotuner) page for derived metrics computation.
