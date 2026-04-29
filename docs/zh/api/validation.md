---
layout: default
title: 输入校验
parent: API 参考
grand_parent: 中文文档
nav_order: 6
description: "输入校验 helper 与运行契约说明"
---

# 输入校验

输入校验 helper 定义在 `triton_ops.validation` 中。它们适合被更高层 wrapper、测试代码或自定义入口复用。

## 核心 helper

### `validate_rmsnorm_rope_inputs`

检查内容：

- `x`、`weight`、`cos`、`sin` 是否在 CUDA 上
- 所有输入是否位于同一设备
- dtype 是否属于支持的浮点类型
- 是否 contiguous
- `x` 是否为 3D
- `weight.shape == (hidden_dim,)`
- `cos` 是否为 2D 或 4D
- `sin.shape == cos.shape`
- 当自动推导 `num_heads` 时，`hidden_dim % head_dim == 0`

返回值：

```python
(batch_size, seq_len, hidden_dim, head_dim, num_heads)
```

### `validate_gated_mlp_inputs`

检查内容：

- CUDA 放置与同设备要求
- 支持的浮点 dtype
- contiguous
- `x` 必须是 3D
- 两个权重必须都是 2D 且形状完全一致
- 激活函数只能是 `"silu"` 或 `"gelu"`

返回值：

```python
(batch_size, seq_len, hidden_dim, intermediate_dim)
```

### `validate_fp8_gemm_inputs`

检查内容：

- 矩阵和 scale 是否在 CUDA 上
- `a` 与 `b` 是否 contiguous
- 是否为 2D 矩阵
- 内部维度 `K` 是否匹配
- 如果输入已经是预量化 FP8，是否同时提供 scale
- 输出 dtype 是否在校验层接受范围内

返回值：

```python
(M, N, K)
```

### `validate_fp8_quantize_inputs`

检查内容：

- CUDA 放置
- 浮点 dtype 支持
- contiguous
- 如果传入 `scale`，则必须是正的标量 `float32`

## 标量检查 helper

可用函数：

- `validate_positive_dimensions(**dims)`
- `validate_head_dim(head_dim)`
- `validate_eps(eps)`

这些函数在 kernel 入口里被内部调用；如果你要在启动 Triton 前更早失败，也可以直接复用它们。
