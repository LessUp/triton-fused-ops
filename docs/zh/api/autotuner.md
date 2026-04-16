---
layout: default
title: "自动调优 API — Triton Fused Ops"
description: "Triton Fused Ops 自动调优框架 API 参考 - TritonAutoTuner、ConfigCache、TuningResult"
---

# 自动调优 API 参考

本文档提供自动调优框架的详细 API 参考。

---

## 目录

- [概述](#概述)
- [TritonAutoTuner](#tritonautotuner)
- [配置空间](#配置空间)
- [ConfigCache](#configcache)
- [TuningResult](#tuningresult)
- [KernelMetrics](#kernelmetrics)

---

## 概述

自动调优为您的特定 GPU 和问题大小寻找最优的算子配置。不同的 GPU 和问题大小受益于不同的块大小、warp 数和流水线级数。

### 为什么需要自动调优？

- **GPU 差异**: A100、H100 和 RTX 4090 有不同的最优配置
- **问题大小**: 小矩阵和大矩阵受益于不同的块大小
- **内存层次结构**: 最优缓存利用率取决于张量维度

### 快速开始

```python
from triton_ops import TritonAutoTuner, FP8_GEMM_CONFIGS

tuner = TritonAutoTuner(
    kernel_fn=my_kernel,
    config_space=FP8_GEMM_CONFIGS,
    warmup_runs=10,
    benchmark_runs=100,
)

result = tuner.tune(*args, problem_size=(M, N, K), device='cuda')
print(f"最优配置: {result.best_config}")
print(f"延迟: {result.metrics.latency_ms:.3f} ms")
```

---

## TritonAutoTuner

自动算子配置搜索框架。

### 语法

```python
triton_ops.TritonAutoTuner(
    kernel_fn: Callable,
    config_space: Dict[str, List[Any]],
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    cache_dir: Optional[str] = None,
)
```

### 参数

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `kernel_fn` | `Callable` | 要调优的算子函数。必须接受 `**kwargs` 传递配置参数。 |
| `config_space` | `Dict[str, List[Any]]` | 要搜索的配置空间。 |
| `warmup_runs` | `int` | 计时前的预热迭代次数。默认：`10`。 |
| `benchmark_runs` | `int` | 计时迭代次数。默认：`100`。 |
| `cache_dir` | `Optional[str]` | 持久化缓存目录。默认：`None`（仅内存）。 |

### 方法

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

搜索配置空间并返回最优配置。

**参数：**
- `*args`: 传递给算子的参数。
- `problem_size`: 用于缓存的问题维度。
- `device`: 用于缓存的设备名称。
- `kernel_type`: 算子类型标识符。
- `**kwargs`: 其他算子参数。

**返回值：** `TuningResult`，包含最优配置。

**异常：** `TuningFailedError`，如果找不到有效配置。

#### get_cached_config

```python
get_cached_config(
    problem_size: Tuple[int, ...],
    device: str,
    kernel_type: str = "unknown",
) -> Optional[Dict[str, Any]]
```

获取缓存的最优配置，无需重新调优。

#### clear_cache

```python
clear_cache() -> None
```

清除所有缓存的配置。

### 示例

```python
import torch
from triton_ops import TritonAutoTuner, FP8_GEMM_CONFIGS

# 定义自定义配置空间
config_space = {
    "BLOCK_M": [64, 128, 256],
    "BLOCK_N": [64, 128, 256],
    "BLOCK_K": [32, 64],
    "num_warps": [4, 8],
}

# 创建调优器
tuner = TritonAutoTuner(
    kernel_fn=my_fp8_gemm_kernel,
    config_space=config_space,
    warmup_runs=5,
    benchmark_runs=50,
    cache_dir="~/.cache/triton_tuning",
)

# 对特定问题大小进行调优
M, N, K = 4096, 4096, 4096
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

result = tuner.tune(
    a, b,
    problem_size=(M, N, K),
    device=torch.cuda.get_device_name(),
    kernel_type="fp8_gemm",
)

print(f"最优配置: {result.best_config}")
print(f"延迟: {result.metrics.latency_ms:.3f} ms")
```

---

## 配置空间

每种算子类型的预定义配置空间。

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

### 辅助函数

#### generate_configs

```python
from triton_ops.autotuner.configs import generate_configs

# 生成所有组合
all_configs = generate_configs(FP8_GEMM_CONFIGS)
# 返回: [{"BLOCK_M": 64, "BLOCK_N": 64, ...}, ...]
```

#### filter_valid_configs

```python
from triton_ops.autotuner.configs import filter_valid_configs

# 根据问题大小约束过滤
valid = filter_valid_configs(all_configs, M=1024, N=1024, K=1024)
```

#### get_default_config

```python
from triton_ops.autotuner.configs import get_default_config

# 获取算子类型的默认配置
default = get_default_config("fp8_gemm")
# 返回: {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, ...}
```

---

## ConfigCache

调优结果的持久化缓存。

### 语法

```python
triton_ops.ConfigCache(
    cache_dir: Optional[str] = None,
)
```

### 参数

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `cache_dir` | `Optional[str]` | 缓存文件目录。如果为 `None`，仅使用内存。 |

### 方法

#### get

```python
get(
    kernel_type: str,
    problem_size: Tuple[int, ...],
    device: str,
) -> Optional[Dict[str, Any]]
```

获取缓存的配置。

#### set

```python
set(
    kernel_type: str,
    problem_size: Tuple[int, ...],
    device: str,
    config: Dict[str, Any],
) -> None
```

将配置存入缓存。

#### clear

```python
clear() -> None
```

清除所有缓存的配置。

#### get_all_keys

```python
get_all_keys() -> list
```

获取所有缓存键。

### 示例

```python
from triton_ops import ConfigCache

# 创建带持久化存储的缓存
cache = ConfigCache(cache_dir="~/.cache/triton_tuning")

# 存储结果
cache.set(
    kernel_type="fp8_gemm",
    problem_size=(4096, 4096, 4096),
    device="NVIDIA A100",
    config={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
)

# 稍后获取
cached = cache.get(
    kernel_type="fp8_gemm",
    problem_size=(4096, 4096, 4096),
    device="NVIDIA A100",
)
print(f"缓存配置: {cached}")
```

---

## TuningResult

自动调优操作的结果。

### 属性

| 属性 | 类型 | 说明 |
|-----------|------|-------------|
| `best_config` | `Dict[str, Any]` | 找到的最优配置。 |
| `metrics` | `KernelMetrics` | 最优配置的性能指标。 |
| `all_results` | `List[Tuple[Dict, KernelMetrics]]` | 所有测试的配置。 |
| `problem_size` | `Optional[Tuple[int, ...]]` | 调优使用的问题大小。 |
| `device` | `Optional[str]` | 调优使用的设备。 |

### 示例

```python
result = tuner.tune(a, b, problem_size=(M, N, K))

# 获取最优配置
print(f"最优配置: {result.best_config}")

# 获取指标
print(f"延迟: {result.metrics.latency_ms:.3f} ms")
print(f"带宽: {result.metrics.bandwidth_gbps:.1f} GB/s")

# 分析所有结果
for config, metrics in result.all_results:
    print(f"{config}: {metrics.latency_ms:.3f} ms")
```

---

## KernelMetrics

算子执行的性能指标。

### 属性

| 属性 | 类型 | 说明 |
|-----------|------|-------------|
| `latency_ms` | `float` | 执行时间（毫秒）。 |
| `throughput_tflops` | `float` | 计算吞吐量（TFLOPS）。 |
| `bandwidth_gbps` | `float` | 内存带宽（GB/s）。 |
| `bandwidth_utilization` | `float` | 峰值带宽百分比。 |

### 字符串表示

```python
metrics = KernelMetrics(
    latency_ms=0.45,
    throughput_tflops=156.2,
    bandwidth_gbps=890.5,
    bandwidth_utilization=43.7,
)
print(metrics)
# 延迟: 0.450 ms, 吞吐量: 156.20 TFLOPS, 带宽: 890.5 GB/s (43.7%)
```

---

## 指标计算函数

### compute_gemm_metrics

```python
from triton_ops.autotuner.tuner import compute_gemm_metrics

metrics = compute_gemm_metrics(
    M=4096, N=4096, K=4096,
    latency_ms=0.45,
    peak_tflops=312.0,  # A100 FP16 峰值
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

**[⬆ 返回顶部](#自动调优-api-参考)** | **[← 返回 API 索引](./)**

</div>
