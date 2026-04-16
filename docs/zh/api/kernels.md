---
layout: default
title: "核心算子 API — Triton Fused Ops"
description: "Triton Fused Ops 核心算子 API 参考 - RMSNorm+RoPE、Gated MLP、FP8 GEMM"
---

# 核心算子 API 参考

本文档提供核心 Triton 融合算子的详细 API 参考。

---

## 目录

- [fused_rmsnorm_rope](#fused_rmsnorm_rope)
- [fused_gated_mlp](#fused_gated_mlp)
- [fp8_gemm](#fp8_gemm)
- [FusedRMSNormRoPE](#fusedrmsnormrope)
- [FusedGatedMLP](#fusedgatedmlp)
- [FP8Linear](#fp8linear)

---

## fused_rmsnorm_rope

应用融合的 RMSNorm + Rotary Position Embedding 变换。

将 RMSNorm 和 RoPE 合并为单次 kernel 启动，通过消除中间 HBM 写入来减少内存带宽需求。

### 语法

```python
triton_ops.fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
    num_heads: Optional[int] = None,
) -> torch.Tensor
```

### 参数

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `x` | `torch.Tensor` | 输入张量，形状 `[batch, seq_len, hidden_dim]`。必须是 CUDA 张量，类型为 `float16` 或 `bfloat16`。 |
| `weight` | `torch.Tensor` | RMSNorm 权重，形状 `[hidden_dim]`。必须与 `x.dtype` 匹配且在 CUDA 上。 |
| `cos` | `torch.Tensor` | 余弦位置编码，形状 `[seq_len, head_dim]`。从位置索引预计算。 |
| `sin` | `torch.Tensor` | 正弦位置编码，形状 `[seq_len, head_dim]`。从位置索引预计算。 |
| `eps` | `float` | RMSNorm 数值稳定性的小常数。默认：`1e-6`。 |
| `num_heads` | `Optional[int]` | 注意力头数。如果为 `None`，从 `hidden_dim / head_dim` 推断。 |

### 返回值

`torch.Tensor` — 应用 RMSNorm + RoPE 后的输出张量，形状 `[batch, seq_len, hidden_dim]`。

### 异常

| 异常 | 条件 |
|-----------|-----------|
| `DeviceError` | CUDA 不可用或张量不在 CUDA 上。 |
| `ShapeMismatchError` | 张量形状不兼容。 |
| `UnsupportedDtypeError` | 张量数据类型不支持。 |

### 数学公式

```
RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
RoPE: y_rope = y * cos + rotate_half(y) * sin
```

### 示例

```python
import torch
from triton_ops import fused_rmsnorm_rope

# 创建输入
batch, seq_len, hidden_dim, head_dim = 2, 128, 4096, 64
x = torch.randn(batch, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
weight = torch.ones(hidden_dim, device='cuda', dtype=torch.float16)

# 预计算位置编码
positions = torch.arange(seq_len, device='cuda')
freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device='cuda') / head_dim))
angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
cos = torch.cos(angles).to(torch.float16)
sin = torch.sin(angles).to(torch.float16)

# 应用融合算子
output = fused_rmsnorm_rope(x, weight, cos, sin)
```

---

## fused_gated_mlp

应用融合的 Gated MLP 变换（SwiGLU/GeGLU）。

计算：`output = activation(gate_proj(x)) * up_proj(x)`

### 语法

```python
triton_ops.fused_gated_mlp(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    activation: Literal["silu", "gelu"] = "silu",
) -> torch.Tensor
```

### 参数

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `x` | `torch.Tensor` | 输入张量，形状 `[batch, seq_len, hidden_dim]`。 |
| `gate_weight` | `torch.Tensor` | Gate 投影权重，形状 `[intermediate_dim, hidden_dim]`。 |
| `up_weight` | `torch.Tensor` | Up 投影权重，形状 `[intermediate_dim, hidden_dim]`。 |
| `activation` | `Literal["silu", "gelu"]` | 激活函数。`"silu"` 用于 SwiGLU，`"gelu"` 用于 GeGLU。默认：`"silu"`。 |

### 返回值

`torch.Tensor` — 输出张量，形状 `[batch, seq_len, intermediate_dim]`。

### 示例

```python
import torch
from triton_ops import fused_gated_mlp

# LLaMA 风格配置
hidden_dim = 4096
intermediate_dim = 11008  # ~2.67x hidden_dim

x = torch.randn(2, 128, hidden_dim, device='cuda', dtype=torch.float16)
gate_w = torch.randn(intermediate_dim, hidden_dim, device='cuda', dtype=torch.float16)
up_w = torch.randn(intermediate_dim, hidden_dim, device='cuda', dtype=torch.float16)

# SwiGLU
output = fused_gated_mlp(x, gate_w, up_w, activation='silu')
```

---

## fp8_gemm

执行 FP8 量化矩阵乘法。

如果输入尚未是 FP8 格式，将自动量化。

### 语法

```python
triton_ops.fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor
```

### 参数

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `a` | `torch.Tensor` | 第一个矩阵，形状 `[M, K]`。可以是 FP8（`uint8`）或浮点（`float16`、`bfloat16`、`float32`）。 |
| `b` | `torch.Tensor` | 第二个矩阵，形状 `[K, N]`。可以是 FP8（`uint8`）或浮点。 |
| `a_scale` | `Optional[torch.Tensor]` | A 的缩放因子（如果 A 是 FP8 则必需，如果是浮点则计算）。 |
| `b_scale` | `Optional[torch.Tensor]` | B 的缩放因子（如果 B 是 FP8 则必需，如果是浮点则计算）。 |
| `output_dtype` | `torch.dtype` | 输出数据类型。默认：`torch.float16`。 |

### 返回值

`torch.Tensor` — 结果矩阵，形状 `[M, N]`，类型为 `output_dtype`。

### 示例

```python
import torch
from triton_ops import fp8_gemm

# 自动量化
a = torch.randn(1024, 4096, device='cuda', dtype=torch.float16)
b = torch.randn(4096, 2048, device='cuda', dtype=torch.float16)
c = fp8_gemm(a, b)  # 自动量化两个输入

# 预量化输入
from triton_ops import quantize_fp8
a_fp8, a_scale = quantize_fp8(a)
b_fp8, b_scale = quantize_fp8(b)
c = fp8_gemm(a_fp8, b_fp8, a_scale, b_scale)
```

---

## FusedRMSNormRoPE

融合 RMSNorm + RoPE 的 PyTorch `nn.Module` 封装。

### 语法

```python
triton_ops.FusedRMSNormRoPE(
    hidden_dim: int,
    head_dim: int,
    eps: float = 1e-6,
)
```

### 参数

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `hidden_dim` | `int` | 隐藏维度大小。 |
| `head_dim` | `int` | RoPE 头维度。 |
| `eps` | `float` | RMSNorm 的 epsilon。默认：`1e-6`。 |

### 属性

| 属性 | 类型 | 说明 |
|-----------|------|-------------|
| `weight` | `nn.Parameter` | 可学习的 RMSNorm 权重，形状 `[hidden_dim]`。 |

### Forward 方法

```python
forward(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor
```

### 示例

```python
import torch
from triton_ops import FusedRMSNormRoPE

# 创建模块
norm = FusedRMSNormRoPE(hidden_dim=4096, head_dim=64).cuda()

# 前向传播
x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
cos = torch.randn(128, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(128, 64, device='cuda', dtype=torch.float16)
output = norm(x, cos, sin)
```

---

## FusedGatedMLP

融合 Gated MLP 的 PyTorch `nn.Module` 封装。

### 语法

```python
triton_ops.FusedGatedMLP(
    hidden_dim: int,
    intermediate_dim: int,
    activation: Literal["silu", "gelu"] = "silu",
)
```

### 参数

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `hidden_dim` | `int` | 输入隐藏维度。 |
| `intermediate_dim` | `int` | FFN 中间维度。 |
| `activation` | `Literal["silu", "gelu"]` | 激活函数。默认：`"silu"`。 |

### 属性

| 属性 | 类型 | 说明 |
|-----------|------|-------------|
| `gate_weight` | `nn.Parameter` | Gate 投影权重，形状 `[intermediate_dim, hidden_dim]`。 |
| `up_weight` | `nn.Parameter` | Up 投影权重，形状 `[intermediate_dim, hidden_dim]`。 |

### 示例

```python
import torch
from triton_ops import FusedGatedMLP

# LLaMA 风格 MLP
mlp = FusedGatedMLP(
    hidden_dim=4096,
    intermediate_dim=11008,
    activation='silu'
).cuda().half()

x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
output = mlp(x)
```

---

## FP8Linear

FP8 量化权重的线性层。

### 语法

```python
triton_ops.FP8Linear(
    in_features: int,
    out_features: int,
    bias: bool = False,
)
```

### 参数

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `in_features` | `int` | 输入特征维度。 |
| `out_features` | `int` | 输出特征维度。 |
| `bias` | `bool` | 是否包含偏置。默认：`False`。 |

### 属性

| 属性 | 类型 | 说明 |
|-----------|------|-------------|
| `weight` | `nn.Parameter` | FP16 权重，首次前向时量化为 FP8。 |
| `weight_fp8` | `Tensor` | FP8 量化权重（延迟初始化）。 |
| `weight_scale` | `Tensor` | FP8 权重的缩放因子。 |

### 示例

```python
import torch
from triton_ops import FP8Linear

# 创建 FP8 线性层
linear = FP8Linear(in_features=4096, out_features=4096).cuda()

# 前向传播（权重在首次调用时量化）
x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
output = linear(x)
```

---

<div align="center">

**[⬆ 返回顶部](#核心算子-api-参考)** | **[← 返回 API 索引](./)**

</div>
