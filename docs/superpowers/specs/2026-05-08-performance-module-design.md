# Performance Module Design

## Summary

Deepen the seam between **Benchmarking** and **Auto-Tuning** by introducing a top-level `triton_ops.performance` module. This module owns derived **Performance metrics** so `BenchmarkSuite` no longer imports metric helpers from `triton_ops.autotuner.tuner`, while `TritonAutoTuner` stays latency-focused.

## Problem

Today, `triton_ops.benchmark.suite` imports `compute_elementwise_metrics` and `compute_gemm_metrics` from `triton_ops.autotuner.tuner`. That makes the benchmark module depend on autotuner implementation details even though the repository documentation describes **Benchmarking** and **Auto-Tuning** as separate support tooling.

The current seam is shallow:

- `Benchmarking` reaches across the seam into `Auto-Tuning`
- formula changes leak across two modules
- tests have no dedicated surface for metric computation logic
- the autotuner implementation carries logic that is not part of its core latency-search job

## Goals

- Give derived **Performance metrics** one deep module with a small interface
- Preserve a latency-first autotuner interface
- Keep `TuningResult` compatible with `KernelMetrics`
- Make direct tests target the new performance module
- Reduce documentation and import-path drift between support modules

## Non-goals

- Changing kernel behavior
- Introducing runtime auto-tuning into the public kernel APIs
- Designing a plugin or registry system for arbitrary metric models
- Revisiting unrelated module splits in this repository

## Chosen design

Create `triton_ops.performance` with a trimmed common-case interface:

```python
from triton_ops import performance

perf = performance.latency_only()
perf = performance.elementwise(
    numel=...,
    bytes_per_element=2,
    peak_bandwidth_gbps=2039.0,
)
perf = performance.gemm(
    M=...,
    N=...,
    K=...,
    bytes_per_element=2,
    peak_tflops=312.0,
    peak_bandwidth_gbps=2039.0,
)

metrics = perf.metrics(latency_ms)
```

Each constructor returns an immutable value object that captures problem-shape context once and later derives `KernelMetrics` from measured latency.

### Why this shape

This design keeps the module deep without widening the public interface:

- simpler than a registry/profile system
- more ergonomic for common callers than a single `derive_metrics(latency_ms, shape=...)` function
- aligned with the repository's small, explicit support-tooling style

## Module responsibilities

`triton_ops.performance` owns:

- validating metric inputs and assumptions
- normalizing zero latency to the existing minimum epsilon behavior
- deriving throughput and bandwidth from latency plus shape context
- constructing `KernelMetrics`

It does not own:

- latency measurement loops
- report formatting
- tuning search and cache logic

## Caller flow

### Benchmarking

`BenchmarkSuite` creates a performance object where problem-shape context is already available, measures latency, and calls `perf.metrics(latency_ms)`.

This makes the common path trivial and keeps formula logic out of the benchmark suite implementation.

### Auto-Tuning

`TritonAutoTuner` remains latency-first, but `tune(...)` may accept an optional prepared performance object from callers that have enough shape context.

When that optional object is provided, the autotuner still ranks configurations by latency only and enriches only `TuningResult.metrics` for the winning configuration. `all_results` stay latency-only in this design to avoid widening the tuning surface further.

## Error handling

- Non-positive dimensions or invalid assumptions raise `ValueError`
- Unsupported constructor shapes raise `ValueError`
- Negative or non-finite latency raises `ValueError`
- Zero latency is normalized internally to the existing minimum-latency epsilon rather than becoming a new public error mode

## Testing strategy

The new performance module becomes the primary test surface for metric formulas.

- Add direct tests for `latency_only`, `elementwise`, and `gemm`
- Keep benchmark tests focused on wiring and observable integration behavior
- Keep autotuner tests focused on tuning behavior and the single optional enrichment path for `TuningResult.metrics`
- Do not duplicate formula assertions across benchmark and autotuner tests

## Alternatives considered

### 1. Minimal function-only interface

Use a single `derive_metrics(latency_ms, shape=None)` entry point plus a `PerformanceShape` value.

**Why not chosen:** it is compact, but it pushes more assembly work to common callers and gives up some leverage from validating shape once.

### 2. Registry/profile interface

Use a flexible `profile()` or `register()` system with pluggable implementations.

**Why not chosen:** the interface is wider than the current repository needs and would likely create a shallow module optimized for hypothetical extensions.

## Consequences

### Positive

- Better locality for formula changes
- Cleaner seam between support modules
- More credible docs because ownership is easier to describe
- Direct, durable tests for metric logic

### Trade-offs

- One more top-level module in the package
- Callers still need correct problem-shape context for derived metrics
- `KernelMetrics` still mixes latency with derived fields, so the model stays slightly broader than the autotuner's core concern

## Follow-up work

1. Implement `triton_ops.performance`
2. Move derived metric logic out of `triton_ops.autotuner.tuner`
3. Rewire `BenchmarkSuite` to depend on the new seam
4. Update benchmark/autotuner docs to match the new ownership
