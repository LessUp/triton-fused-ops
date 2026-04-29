---
layout: default
title: Auto-Tuning
parent: API Reference
grand_parent: Documentation
nav_order: 3
description: "Generic configuration search, caching, and metric helpers"
---

# Auto-Tuning

The autotuning layer in this repository is generic infrastructure for user-supplied callables. The shipped kernels do not automatically run this tuner inside their public API.

## `TritonAutoTuner`

```python
TritonAutoTuner(
    kernel_fn: Callable,
    config_space: dict[str, list[Any]],
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    cache_dir: str | None = None,
)
```

Key expectations:

- `kernel_fn` must accept the searched configuration values as keyword arguments.
- The tuner repeatedly benchmarks each configuration and keeps the lowest-latency result.
- If `problem_size` and `device` are provided to `tune`, the best config can be cached.

Example:

```python
import torch
from triton_ops import TritonAutoTuner

def dummy_kernel(x, BLOCK_SIZE=64, num_warps=4):
    return x * 2

tuner = TritonAutoTuner(
    kernel_fn=dummy_kernel,
    config_space={
        "BLOCK_SIZE": [64, 128],
        "num_warps": [4, 8],
    },
    warmup_runs=2,
    benchmark_runs=5,
)

x = torch.randn(1024, device="cuda")
result = tuner.tune(x, problem_size=(1024,), device="cuda:0", kernel_type="dummy")
```

## `ConfigCache`

```python
ConfigCache(cache_dir: str | None = None)
```

Cache key dimensions:

- `kernel_type`
- `problem_size`
- `device`

Behavior:

- Always keeps an in-memory cache.
- If `cache_dir` is set, also persists JSON files.
- Uses thread-safe access around the in-memory store.

Methods:

- `get(kernel_type, problem_size, device)`
- `set(kernel_type, problem_size, device, config)`
- `clear()`
- `get_all_keys()`

## Configuration spaces

The repository exports three pre-defined config spaces:

- `RMSNORM_ROPE_CONFIGS`
- `GATED_MLP_CONFIGS`
- `FP8_GEMM_CONFIGS`

These are plain dictionaries from parameter names to candidate value lists. They live in `triton_ops.autotuner.configs`.

Helper functions in the same module:

- `generate_configs(config_space)`
- `filter_valid_configs(configs, hidden_dim=None, intermediate_dim=None, M=None, N=None, K=None)`
- `get_default_config(kernel_type)`

## Result and metric objects

The tuning API returns `TuningResult`, which contains:

- `best_config`
- `metrics`
- `all_results`
- `problem_size`
- `device`

Metrics are represented by `KernelMetrics`.

Metric helpers in `triton_ops.autotuner.tuner`:

- `compute_gemm_metrics(M, N, K, latency_ms, ...)`
- `compute_elementwise_metrics(numel, latency_ms, ...)`

## Failure mode

If no configuration succeeds, the tuner raises `TuningFailedError`.

## Practical guidance

- Use autotuning when you are experimenting with custom Triton kernels or wrappers.
- Do not assume that the public `fused_rmsnorm_rope`, `fused_gated_mlp`, or `fp8_gemm` functions consult the tuner automatically.
- Treat cache keys as part of your experiment design: the same kernel family on a different device string becomes a separate cache entry.
