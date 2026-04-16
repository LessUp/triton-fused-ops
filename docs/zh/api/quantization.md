---
layout: default
title: "FP8 量化 API — Triton Fused Ops"
description: "Triton Fused Ops FP8 量化 API 参考 - quantize_fp8、dequantize_fp8、FP8Format"
---

# FP8 量化 API 参考

本文档提供 FP8 量化工具的详细 API 参考。

---

## 目录

- [概述](#概述)
- [quantize_fp8](#quantize_fp8)
- [dequantize_fp8](#dequantize_fp8)
- [quantize_fp8_with_overflow_handling](#quantize_fp8_with_overflow_handling)
- [FP8Format](#fp8format)

---

## 概述

FP8（8 位浮点）量化相比 FP16 可减少 50% 的内存使用，同时在推理中保持精度。本库使用 **E4M3** 格式：

| 属性 | 值 |
|----------|-------|
| 符号位 | 1 |
| 指数位 | 4 |
| 尾数位 | 3 |
| 最大值 | 448.0 |
| 最小规约数 | 2^-6 ≈ 0.015625 |

### 存储格式

FP8 值以 `uint8` 张量形式存储以确保兼容性：

```python
# FP8 存储为 uint8
quantized: torch.uint8  # shape [...]
scale: torch.float32    # shape [1]
```

---

## quantize_fp8

将张量量化为 FP8 E4M3 格式。

### 语法

```python
triton_ops.quantize_fp8(
    tensor: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]
```

### 参数

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `tensor` | `torch.Tensor` | 输入张量，类型为 `float16`、`bfloat16` 或 `float32`。必须在 CUDA 上。 |
| `scale` | `Optional[torch.Tensor]` | 预计算的缩放因子。如果为 `None`，自动计算。 |

### 返回值

`tuple[torch.Tensor, torch.Tensor]`：
- **quantized**：FP8 值，类型为 `uint8`，形状与输入相同。
- **scale**：量化使用的缩放因子，形状 `[1]`，类型 `float32`。

### 缩放计算

如果未提供 `scale`，则计算为：

```
scale = FP8_MAX / max(abs(tensor))
      = 448.0 / max(abs(tensor))
```

### 示例

```python
import torch
from triton_ops import quantize_fp8

# 创建输入张量
tensor = torch.randn(1024, 4096, device='cuda', dtype=torch.float16)
print(f"原始: {tensor.dtype}, {tensor.element_size()} 字节")

# 量化为 FP8
quantized, scale = quantize_fp8(tensor)
print(f"量化后: {quantized.dtype}, {quantized.element_size()} 字节")
print(f"缩放: {scale.item():.6f}")

# 内存节省: 50%
# FP16: 每元素 2 字节
# FP8:  每元素 1 字节
```

---

## dequantize_fp8

将 FP8 张量反量化为浮点格式。

### 语法

```python
triton_ops.dequantize_fp8(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor
```

### 参数

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `tensor` | `torch.Tensor` | 以 `uint8` 存储的 FP8 张量。 |
| `scale` | `torch.Tensor` | 量化时的缩放因子。 |
| `output_dtype` | `torch.dtype` | 输出数据类型。默认：`torch.float16`。 |

### 返回值

`torch.Tensor` — 反量化后的张量，类型为 `output_dtype`。

### 示例

```python
import torch
from triton_ops import quantize_fp8, dequantize_fp8

# 量化
original = torch.randn(512, 512, device='cuda', dtype=torch.float16)
quantized, scale = quantize_fp8(original)

# 反量化
recovered = dequantize_fp8(quantized, scale, output_dtype=torch.float16)

# 检查重建误差
error = torch.abs(original - recovered).mean().item()
print(f"平均重建误差: {error:.6f}")
```

---

## quantize_fp8_with_overflow_handling

量化到 FP8，带动态溢出处理。

如果检测到溢出（缩放后值超出 FP8 范围），则自动调整缩放因子并重试量化。

### 语法

```python
triton_ops.quantize_fp8_with_overflow_handling(
    tensor: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    max_attempts: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]
```

### 参数

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `tensor` | `torch.Tensor` | 输入张量。 |
| `scale` | `Optional[torch.Tensor]` | 初始缩放因子。如果为 `None`，自动计算。 |
| `max_attempts` | `int` | 最大重试次数。默认：`3`。 |

### 返回值

`tuple[torch.Tensor, torch.Tensor]`：
- **quantized**：FP8 值。
- **final_scale**：调整后的缩放因子。

### 异常

| 异常 | 条件 |
|-----------|-----------|
| `NumericalOverflowError` | `max_attempts` 次尝试后仍无法解决溢出。 |

### 示例

```python
import torch
from triton_ops import quantize_fp8_with_overflow_handling

# 可能溢出的张量
tensor = torch.randn(1024, 4096, device='cuda', dtype=torch.float16) * 500

# 带溢出处理的安全量化
try:
    quantized, scale = quantize_fp8_with_overflow_handling(tensor, max_attempts=3)
    print(f"量化成功，缩放: {scale.item():.6f}")
except NumericalOverflowError as e:
    print(f"量化失败: {e}")
```

---

## FP8Format

FP8 E4M3 格式规范和工具的 Dataclass。

### 属性

| 属性 | 类型 | 默认值 | 说明 |
|-----------|------|---------|-------------|
| `exponent_bits` | `int` | `4` | 指数位数。 |
| `mantissa_bits` | `int` | `3` | 尾数位数。 |
| `max_value` | `float` | `448.0` | 最大可表示值。 |
| `min_normal` | `float` | `2**-6` | 最小规约数。 |

### 静态方法

#### compute_scale

```python
FP8Format.compute_scale(tensor: torch.Tensor) -> torch.Tensor
```

计算 FP8 转换的最优缩放因子。

**返回值：** `torch.Tensor` — 缩放因子，形状 `[1]`，类型 `float32`。

#### compute_scale_per_channel

```python
FP8Format.compute_scale_per_channel(
    tensor: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor
```

计算 FP8 转换的逐通道缩放因子。

**返回值：** `torch.Tensor` — 逐通道缩放，形状与输入相同，但减少指定维度。

#### is_in_range

```python
FP8Format.is_in_range(
    tensor: torch.Tensor,
    scale: torch.Tensor,
) -> bool
```

检查缩放后的张量是否在 FP8 可表示范围内。

**返回值：** `bool` — `True` 如果所有值都在 `[-448.0, 448.0]` 内。

### 示例

```python
from triton_ops import FP8Format
import torch

# 格式属性
print(f"FP8 最大值: {FP8Format.max_value}")  # 448.0
print(f"FP8 最小规约数: {FP8Format.min_normal}")  # 0.015625

# 计算缩放
tensor = torch.randn(512, 512, device='cuda', dtype=torch.float16)
scale = FP8Format.compute_scale(tensor)
print(f"缩放: {scale.item():.6f}")

# 检查是否在范围内
if FP8Format.is_in_range(tensor, scale):
    print("张量在缩放后处于 FP8 范围内")
```

---

## 最佳实践

### 1. 使用自动缩放

大多数情况下，让库自动计算缩放：

```python
quantized, scale = quantize_fp8(tensor)  # 自动缩放
```

### 2. 缓存缩放因子

对于重复操作，计算一次缩放并复用：

```python
# 计算一次缩放
_, scale = quantize_fp8(weight)

# 对多个输入复用
for x in inputs:
    x_quantized, _ = quantize_fp8(x, scale=scale)
```

### 3. 检查精度

始终验证您用例的重建误差：

```python
quantized, scale = quantize_fp8(tensor)
recovered = dequantize_fp8(quantized, scale)
error = torch.abs(tensor - recovered).max().item()

if error > 1.0:
    print(f"警告：量化误差较大: {error}")
```

### 4. 处理边界情况

对极值张量使用溢出处理：

```python
from triton_ops import quantize_fp8_with_overflow_handling, NumericalOverflowError

try:
    quantized, scale = quantize_fp8_with_overflow_handling(tensor)
except NumericalOverflowError:
    # 回退到 FP16 或裁剪张量
    pass
```

---

<div align="center">

**[⬆ 返回顶部](#fp8-量化-api-参考)** | **[← 返回 API 索引](./)**

</div>
