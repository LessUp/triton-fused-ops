---
layout: default
title: "Auto-Tuning API — Triton Fused Ops"
description: "API reference for auto-tuning framework - TritonAutoTuner, ConfigCache, TuningResult"
---

# Auto-Tuning API Reference

This document provides detailed API reference for the auto-tuning framework.

---

## Table of Contents

- [Overview](#overview)
- [TritonAutoTuner](#tritonautotuner)
- [Configuration Spaces](#configuration-spaces)
- [ConfigCache](#configcache)
- [TuningResult](#tuningresult)
- [KernelMetrics](#kernelmetrics)

---

## Overview

Auto-tuning finds optimal kernel configurations for your specific GPU and problem size. Different GPUs and problem sizes benefit from different block sizes, warp counts, and pipeline stages.

### Why Auto-Tuning?

- **GPU Variability**: A100, H100, and RTX 4090 have different optimal configurations
- **Problem Size**: Small and large matrices benefit from different block sizes
- **Memory Hierarchy**: Optimal cache utilization depends on tensor dimensions

### Quick Start

```python
from triton_ops import TritonAutoTuner, FP8_GEMM_CONFIGS

tuner = TritonAutoTuner(
    kernel_fn=my_kernel,
    config_space=FP8_GEMM_CONFIGS,
    warmup_runs=10,
    benchmark_runs=100,
)

result = tuner.tune(*args, problem_size=(M, N, K), device='cuda')
print(f"Best config: {result.best_config}")
print(f"Latency: {result.metrics.latency_ms:.3f} ms")
```

---

## TritonAutoTuner

Automatic kernel configuration search framework.

### Syntax

```python
triton_ops.TritonAutoTuner(
    kernel_fn: Callable,
    config_space: Dict[str, List[Any]],
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    cache_dir: Optional[str] = None,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `kernel_fn` | `Callable` | Kernel function to tune. Must accept `**kwargs` for config parameters. |
| `config_space` | `Dict[str, List[Any]]` | Configuration space to search. |
| `warmup_runs` | `int` | Warmup iterations before timing. Default: `10`. |
| `benchmark_runs` | `int` | Timing iterations. Default: `100`. |
| `cache_dir` | `Optional[str]` | Directory for persistent cache. Default: `None` (memory only). |

### Methods

#### tune

```python
tune(
    *args,
    problem_size: Tuple[int, ...] = None,
    device: str = None,
    kernel_type: str = "unknown",
    **kwargs,
) -> TuningResult
```

Search configuration space and return optimal config.

**Parameters:**
- `*args`: Arguments to pass to kernel.
- `problem_size`: Problem dimensions for caching.
- `device`: Device name for caching.
- `kernel_type`: Kernel type identifier.
- `**kwargs`: Additional kernel arguments.

**Returns:** `TuningResult` with best configuration.

**Raises:** `TuningFailedError` if no valid configuration found.

#### get_cached_config

```python
get_cached_config(
    problem_size: Tuple[int, ...],
    device: str,
    kernel_type: str = "unknown",
) -> Optional[Dict[str, Any]]
```

Retrieve cached optimal configuration without re-tuning.

#### clear_cache

```python
clear_cache() -> None
```

Clear all cached configurations.

### Example

```python
import torch
from triton_ops import TritonAutoTuner, FP8_GEMM_CONFIGS

# Define custom config space
config_space = {
    "BLOCK_M": [64, 128, 256],
    "BLOCK_N": [64, 128, 256],
    "BLOCK_K": [32, 64],
    "num_warps": [4, 8],
}

# Create tuner
tuner = TritonAutoTuner(
    kernel_fn=my_fp8_gemm_kernel,
    config_space=config_space,
    warmup_runs=5,
    benchmark_runs=50,
    cache_dir="~/.cache/triton_tuning",
)

# Tune for specific problem size
M, N, K = 4096, 4096, 4096
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

result = tuner.tune(
    a, b,
    problem_size=(M, N, K),
    device=torch.cuda.get_device_name(),
    kernel_type="fp8_gemm",
)

print(f"Best config: {result.best_config}")
print(f"Latency: {result.metrics.latency_ms:.3f} ms")
```

---

## Configuration Spaces

Pre-defined configuration spaces for each kernel type.

### RMSNORM_ROPE_CONFIGS

```python
RMSNORM_ROPE_CONFIGS = {
    "BLOCK_SIZE": [64, 128, 256, 512, 1024],
    "num_warps": [2, 4, 8],
    "num_stages": [1, 2, 3],
}
```

### GATED_MLP_CONFIGS

```python
GATED_MLP_CONFIGS = {
    "BLOCK_M": [32, 64, 128],
    "BLOCK_N": [32, 64, 128],
    "BLOCK_K": [32, 64],
    "num_warps": [4, 8],
    "num_stages": [2, 3, 4],
}
```

### FP8_GEMM_CONFIGS

```python
FP8_GEMM_CONFIGS = {
    "BLOCK_M": [64, 128, 256],
    "BLOCK_N": [64, 128, 256],
    "BLOCK_K": [32, 64],
    "GROUP_SIZE_M": [4, 8],
    "num_warps": [4, 8],
    "num_stages": [3, 4, 5],
}
```

### Helper Functions

#### generate_configs

```python
from triton_ops.autotuner.configs import generate_configs

# Generate all combinations
all_configs = generate_configs(FP8_GEMM_CONFIGS)
# Returns: [{"BLOCK_M": 64, "BLOCK_N": 64, ...}, ...]
```

#### filter_valid_configs

```python
from triton_ops.autotuner.configs import filter_valid_configs

# Filter for problem size constraints
valid = filter_valid_configs(all_configs, M=1024, N=1024, K=1024)
```

#### get_default_config

```python
from triton_ops.autotuner.configs import get_default_config

# Get default config for kernel type
default = get_default_config("fp8_gemm")
# Returns: {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, ...}
```

---

## ConfigCache

Persistent cache for tuning results.

### Syntax

```python
triton_ops.ConfigCache(
    cache_dir: Optional[str] = None,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `cache_dir` | `Optional[str]` | Directory for cache files. If `None`, uses memory only. |

### Methods

#### get

```python
get(
    kernel_type: str,
    problem_size: Tuple[int, ...],
    device: str,
) -> Optional[Dict[str, Any]]
```

Retrieve cached configuration.

#### set

```python
set(
    kernel_type: str,
    problem_size: Tuple[int, ...],
    device: str,
    config: Dict[str, Any],
) -> None
```

Store configuration in cache.

#### clear

```python
clear() -> None
```

Clear all cached configurations.

#### get_all_keys

```python
get_all_keys() -> list
```

Get all cache keys.

### Example

```python
from triton_ops import ConfigCache

# Create cache with persistent storage
cache = ConfigCache(cache_dir="~/.cache/triton_tuning")

# Store result
cache.set(
    kernel_type="fp8_gemm",
    problem_size=(4096, 4096, 4096),
    device="NVIDIA A100",
    config={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
)

# Retrieve later
cached = cache.get(
    kernel_type="fp8_gemm",
    problem_size=(4096, 4096, 4096),
    device="NVIDIA A100",
)
print(f"Cached config: {cached}")
```

---

## TuningResult

Result from auto-tuning operation.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `best_config` | `Dict[str, Any]` | Optimal configuration found. |
| `metrics` | `KernelMetrics` | Performance metrics for best config. |
| `all_results` | `List[Tuple[Dict, KernelMetrics]]` | All tested configurations. |
| `problem_size` | `Optional[Tuple[int, ...]]` | Problem size used for tuning. |
| `device` | `Optional[str]` | Device used for tuning. |

### Example

```python
result = tuner.tune(a, b, problem_size=(M, N, K))

# Access best config
print(f"Best config: {result.best_config}")

# Access metrics
print(f"Latency: {result.metrics.latency_ms:.3f} ms")
print(f"Bandwidth: {result.metrics.bandwidth_gbps:.1f} GB/s")

# Analyze all results
for config, metrics in result.all_results:
    print(f"{config}: {metrics.latency_ms:.3f} ms")
```

---

## KernelMetrics

Performance metrics for kernel execution.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `latency_ms` | `float` | Execution time in milliseconds. |
| `throughput_tflops` | `float` | Computational throughput in TFLOPS. |
| `bandwidth_gbps` | `float` | Memory bandwidth in GB/s. |
| `bandwidth_utilization` | `float` | Percentage of peak bandwidth. |

### String Representation

```python
metrics = KernelMetrics(
    latency_ms=0.45,
    throughput_tflops=156.2,
    bandwidth_gbps=890.5,
    bandwidth_utilization=43.7,
)
print(metrics)
# Latency: 0.450 ms, Throughput: 156.20 TFLOPS, Bandwidth: 890.5 GB/s (43.7%)
```

---

## Metric Computation Functions

### compute_gemm_metrics

```python
from triton_ops.autotuner.tuner import compute_gemm_metrics

metrics = compute_gemm_metrics(
    M=4096, N=4096, K=4096,
    latency_ms=0.45,
    peak_tflops=312.0,  # A100 FP16 peak
    peak_bandwidth_gbps=2039.0,  # A100 HBM
)
```

### compute_elementwise_metrics

```python
from triton_ops.autotuner.tuner import compute_elementwise_metrics

metrics = compute_elementwise_metrics(
    numel=4096 * 4096,
    latency_ms=0.1,
    bytes_per_element=2,  # FP16
    peak_bandwidth_gbps=2039.0,
)
```

---

<div align="center">

**[⬆ Back to Top](#auto-tuning-api-reference)** | **[← Back to API Index](./)**

</div>
