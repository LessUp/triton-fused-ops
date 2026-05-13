---
title: 自动调优
description: "通用配置搜索、缓存与性能指标 helper"
---

# 自动调优

本仓库的自动调优层是面向“用户自定义 callable”的通用基础设施。仓库导出的 kernel 入口函数不会在正常调用时自动走这套 tuner。

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

关键约束：

- `kernel_fn` 必须能够接受被搜索配置作为关键字参数。
- tuner 会逐个 benchmark 配置，并保留延迟最低的结果。
- 如果在 `tune` 中传入 `problem_size` 与 `device`，结果就可以缓存。

## `ConfigCache`

```python
ConfigCache(cache_dir: str | None = None)
```

缓存键维度：

- `kernel_type`
- `problem_size`
- `device`

行为：

- 始终维护一份内存缓存。
- 如果设置 `cache_dir`，同时会持久化成 JSON 文件。
- 内存层使用线程安全访问。

常用方法：

- `get(kernel_type, problem_size, device)`
- `set(kernel_type, problem_size, device, config)`
- `clear()`
- `get_all_keys()`

## 配置空间

仓库导出三套预定义配置空间：

- `RMSNORM_ROPE_CONFIGS`
- `GATED_MLP_CONFIGS`
- `FP8_GEMM_CONFIGS`

这些对象本质上都是“参数名 -> 候选值列表”的普通字典，定义在 `triton_ops.autotuner.configs`。

同模块还提供：

- `generate_configs(config_space)`
- `filter_valid_configs(configs, hidden_dim=None, intermediate_dim=None, M=None, N=None, K=None)`
- `get_default_config(kernel_type)`

## 结果对象与指标对象

调优结果使用 `TuningResult` 表示，里面包含：

- `best_config`
- `metrics`
- `all_results`
- `problem_size`
- `device`

性能指标统一由 `KernelMetrics` 表示。

`triton_ops.autotuner.tuner` 还提供两个常用指标 helper：

- `compute_gemm_metrics(M, N, K, latency_ms, ...)`
- `compute_elementwise_metrics(numel, latency_ms, ...)`

## 失败模式

如果所有配置都失败，tuner 会抛出 `TuningFailedError`。

## 使用建议

- 当你在实验自定义 Triton kernel 或 wrapper 时，自动调优很有价值。
- 不要假设 `fused_rmsnorm_rope`、`fused_gated_mlp` 或 `fp8_gemm` 在正常调用时会自动查询 tuner。
- 把缓存键当成实验设计的一部分：设备字符串变了，就会形成新的缓存项。
