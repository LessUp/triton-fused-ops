---
title: 快速开始
description: "函数式 API 与模块封装的第一批可运行示例"
---

# 快速开始

本页给出当前公开 API 的最短可运行路径。

## 根包导入

```python
import torch
from triton_ops import (
    fused_rmsnorm_rope,
    fused_gated_mlp,
    fp8_gemm,
    quantize_fp8,
    FusedRMSNormRoPE,
    FusedGatedMLP,
    FP8Linear,
)
```

## `fused_rmsnorm_rope`

```python
import torch
from triton_ops import fused_rmsnorm_rope

batch, seq_len, hidden_dim, head_dim = 2, 128, 4096, 64
x = torch.randn(batch, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float16)
cos = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
sin = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)

y = fused_rmsnorm_rope(x, weight, cos, sin)
print(y.shape)
```

要点：

- `x` 必须是 3D 且 contiguous。
- `weight.shape` 必须等于 `(hidden_dim,)`。
- `cos` 和 `sin` 的形状必须一致。
- 如果不传 `num_heads`，实现会自动按 `hidden_dim / head_dim` 推断。

## `fused_gated_mlp`

```python
import torch
from triton_ops import fused_gated_mlp

hidden_dim = 4096
intermediate_dim = 11008

x = torch.randn(2, 128, hidden_dim, device="cuda", dtype=torch.float16)
gate_weight = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16)
up_weight = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16)

y = fused_gated_mlp(x, gate_weight, up_weight, activation="silu")
print(y.shape)
```

当前 kernel 与参考实现都遵循：

```text
output = activation(gate_proj(x)) * up_proj(x)
```

支持的激活函数只有 `"silu"` 和 `"gelu"`。

## `fp8_gemm`

```python
import torch
from triton_ops import fp8_gemm, quantize_fp8

a = torch.randn(1024, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 2048, device="cuda", dtype=torch.float16)

# 方式 1：让库自动量化
c_auto = fp8_gemm(a, b)

# 方式 2：显式量化后传入 scale
a_fp8, a_scale = quantize_fp8(a)
b_fp8, b_scale = quantize_fp8(b)
c_manual = fp8_gemm(a_fp8, b_fp8, a_scale, b_scale)

print(c_auto.shape, c_manual.shape)
```

实际使用时，请优先把输出视为 `torch.float16` 或 `torch.bfloat16` 路径。这是当前 Triton kernel 真正维护的半精度输出路径。

## 模块封装

```python
import torch
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP, FP8Linear

class DecoderBlock(torch.nn.Module):
    def __init__(self, hidden_dim=4096, num_heads=32, intermediate_dim=11008):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.norm = FusedRMSNormRoPE(hidden_dim, head_dim)
        self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim, activation="silu")
        self.proj = FP8Linear(hidden_dim, hidden_dim)

    def forward(self, x, cos, sin):
        normed = self.norm(x, cos, sin)
        mixed = self.proj(normed)
        mlp_out = self.mlp(normed)
        return mixed, mlp_out
```

`FP8Linear` 会在第一次前向时延迟量化权重，并缓存转置后的 FP8 权重副本。

## 下一步

- [示例教程](/zh/getting-started/examples)
- [集成指南](/zh/guides/integration)
- [核心算子 API](/zh/api/kernels)
