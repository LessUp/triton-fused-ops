---
title: FP8 量化
description: "FP8 存储、scale 规则与反量化 helper"
---

# FP8 量化

本仓库实现的是一种以 `uint8` 存储加显式 scale 为核心的 FP8 兼容路径。

## 格式模型

代码中的核心常量与语义包括：

- `FP8_MAX = 448.0`
- 当前内置运行路径采用每个 tensor 一个标量 scale
- 通过带偏移量的 `uint8` 存储来表示有符号范围

范围提醒：

- 校验层会在 PyTorch 支持原生 float8 dtype 时识别它们。
- 但当前真正维护的 kernel 路径，仍然是基于 `uint8` 兼容表示来量化与解释数据。

## `quantize_fp8`

```python
quantize_fp8(
    tensor: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

要求：

- `tensor` 必须是 contiguous 的 CUDA 张量。
- 支持的输入 dtype：`float16`、`bfloat16`、`float32`。
- 如果传入 `scale`，它必须是正的 CUDA 标量，dtype 为 `float32`。

返回：

- dtype 为 `torch.uint8` 的量化结果
- dtype 为 `torch.float32` 的 scale

自动 scale 规则：

```text
scale = 448.0 / max(abs(tensor))
```

如果输入 tensor 全为零，scale 会返回 `1.0`。

## `dequantize_fp8`

```python
dequantize_fp8(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor
```

该函数用于把仓库定义的 FP8 存储格式还原回浮点张量。

## 带溢出处理的 helper

这个 helper 实现在 kernel 模块里，并没有从 `triton_ops.__init__` 重新导出。

导入路径：

```python
from triton_ops.kernels.fp8_quantize import quantize_fp8_with_overflow_handling
```

签名：

```python
quantize_fp8_with_overflow_handling(
    tensor: torch.Tensor,
    scale: torch.Tensor | None = None,
    max_attempts: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]
```

行为：

- 先检查 `tensor * scale` 是否还在 FP8 范围内。
- 若超界，则把 scale 减半后重试。
- 如果多次重试仍失败，则抛出 `NumericalOverflowError`。

## `FP8Format`

`FP8Format` 定义在 `triton_ops.models` 中，同时也从根包导出。

常用成员：

- `FP8Format.max_value`
- `FP8Format.min_normal`
- `FP8Format.compute_scale(tensor)`
- `FP8Format.compute_scale_per_channel(tensor, dim=0)`
- `FP8Format.is_in_range(tensor, scale)`

实践提醒：

- 当前内置 `fp8_gemm` 路径消费的是标量 scale，而不是逐通道 scale。
- `compute_scale_per_channel` 仍然适合你在仓库外自定义更复杂的量化流程时使用。

## 推荐使用方式

- 想显式控制量化流程时，使用 `quantize_fp8` + `fp8_gemm`。
- 想快速上手时，直接用 `fp8_gemm(a, b)` 让库自动量化。
- 对数值敏感的归一化边界，建议继续保持更高精度。
- 在自己的实际 workload 上，一定要对比 FP16 baseline 再决定是否扩大 FP8 覆盖面。
