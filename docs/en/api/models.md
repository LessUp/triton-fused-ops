---
title: Models and Types
description: "Dataclasses and result containers used across validation, benchmarking, and autotuning"
---

# Models and Types

The `triton_ops.models` module contains the repository's shared dataclasses for input specification, metrics, tuning results, and FP8 format utilities.

## `TensorSpec`

```python
TensorSpec(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str = "cuda",
    contiguous: bool = True,
)
```

Key methods:

- `validate(tensor) -> bool`
- `create_tensor(fill_value=None) -> torch.Tensor`

Use `TensorSpec` when you want a declarative description of an expected tensor.

## Input-spec dataclasses

Available builders:

- `RMSNormRoPEInput.from_shapes(...)`
- `GatedMLPInput.from_shapes(...)`
- `FP8GEMMInput.from_shapes(...)`

These classes package related `TensorSpec` instances for tests, scaffolding, and examples.

Important note on `FP8GEMMInput`:

- It chooses a float8 dtype when PyTorch exposes one, otherwise `uint8`.
- The maintained runtime kernel path in this repository still uses the `uint8` compatibility format described in the quantization page.

## `KernelMetrics`

```python
KernelMetrics(
    latency_ms: float,
    throughput_tflops: float,
    bandwidth_gbps: float,
    bandwidth_utilization: float,
)
```

This is the common metric container used by the benchmark and autotuning layers.

## `TuningResult`

```python
TuningResult(
    best_config: dict[str, Any],
    metrics: KernelMetrics,
    all_results: list[tuple[dict[str, Any], KernelMetrics]] = [],
    problem_size: tuple[int, ...] | None = None,
    device: str | None = None,
)
```

It records the best configuration, its metrics, and optionally the full search results.

## `FP8Format`

`FP8Format` stores the FP8 E4M3-related constants and utility methods used throughout the quantization code.

Common members:

- `FP8Format.max_value`
- `FP8Format.min_normal`
- `FP8Format.compute_scale(tensor)`
- `FP8Format.compute_scale_per_channel(tensor, dim=0)`
- `FP8Format.is_in_range(tensor, scale)`
