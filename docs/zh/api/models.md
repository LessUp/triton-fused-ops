---
title: 数据模型与类型
description: "校验、benchmark 与自动调优中复用的 dataclass 与结果容器"
---

# 数据模型与类型

`triton_ops.models` 模块集中放置了输入规格、性能指标、调优结果以及 FP8 格式工具等 dataclass。

## `TensorSpec`

```python
TensorSpec(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: str = "cuda",
    contiguous: bool = True,
)
```

核心方法：

- `validate(tensor) -> bool`
- `create_tensor(fill_value=None) -> torch.Tensor`

## 输入规格 dataclass

可用构造器：

- `RMSNormRoPEInput.from_shapes(...)`
- `GatedMLPInput.from_shapes(...)`
- `FP8GEMMInput.from_shapes(...)`

这些类很适合用于测试、脚手架和样例生成。

关于 `FP8GEMMInput` 的提醒：

- 当 PyTorch 支持原生 float8 dtype 时，它会优先选择 float8。
- 但仓库当前维护的运行路径仍然是量化页中描述的 `uint8` 兼容表示。

## `KernelMetrics`

```python
KernelMetrics(
    latency_ms: float,
    throughput_tflops: float,
    bandwidth_gbps: float,
    bandwidth_utilization: float,
)
```

这是 benchmark 与 autotuning 共享的性能指标容器。

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

用于保存最优配置、指标以及可选的完整搜索结果。

## `FP8Format`

`FP8Format` 保存了 FP8 E4M3 相关的常量和工具函数。

常用成员：

- `FP8Format.max_value`
- `FP8Format.min_normal`
- `FP8Format.compute_scale(tensor)`
- `FP8Format.compute_scale_per_channel(tensor, dim=0)`
- `FP8Format.is_in_range(tensor, scale)`
