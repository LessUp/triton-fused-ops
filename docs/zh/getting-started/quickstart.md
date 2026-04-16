---
layout: default
title: "快速开始 — Triton Fused Ops"
description: "5 分钟快速上手 Triton Fused Ops - 您的第一个融合算子"
---

# 🚀 快速开始指南

在 5 分钟内开始使用 Triton Fused Ops。

---

## ⚡ 3 行代码集成

用融合算子替换现有的 PyTorch 操作：

```python
import torch
from triton_ops import fused_rmsnorm_rope, fused_gated_mlp, fp8_gemm
```

---

## 🎯 示例 1：融合 RMSNorm + RoPE

### 之前（分离操作）

```python
# 标准 PyTorch - 2 次 kernel 启动，中间结果写入 HBM
x_norm = rmsnorm(x)      # Kernel 1: 加载 x，计算 norm，写入 x_norm
output = rope(x_norm)    # Kernel 2: 加载 x_norm，计算 RoPE，写入 output
# 总计：每个元素 3 次 HBM 读，2 次 HBM 写
```

### 之后（融合）

```python
from triton_ops import fused_rmsnorm_rope

# 单次融合 kernel - 无中间 HBM 写入
output = fused_rmsnorm_rope(x, weight, cos, sin)
# 总计：每个元素 1 次 HBM 读，1 次 HBM 写
```

### 完整示例

```python
import torch
from triton_ops import fused_rmsnorm_rope

# 准备输入
batch, seq_len, hidden_dim = 2, 128, 4096
head_dim = 64

x = torch.randn(batch, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
weight = torch.ones(hidden_dim, device='cuda', dtype=torch.float16)

# 预计算 RoPE 位置编码
positions = torch.arange(seq_len, device='cuda')
freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device='cuda') / head_dim))
angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
cos = torch.cos(angles).to(torch.float16)
sin = torch.sin(angles).to(torch.float16)

# 应用融合算子
output = fused_rmsnorm_rope(x, weight, cos, sin)
print(f"输出形状: {output.shape}")  # [2, 128, 4096]
```

---

## 🎯 示例 2：融合 Gated MLP (SwiGLU)

```python
import torch
from triton_ops import fused_gated_mlp

# LLaMA 风格配置
hidden_dim = 4096
intermediate_dim = 11008  # ~2.67x hidden_dim

x = torch.randn(2, 128, hidden_dim, device='cuda', dtype=torch.float16)
gate_w = torch.randn(intermediate_dim, hidden_dim, device='cuda', dtype=torch.float16)
up_w = torch.randn(intermediate_dim, hidden_dim, device='cuda', dtype=torch.float16)

# 单次融合 kernel 完成 SwiGLU
output = fused_gated_mlp(x, gate_w, up_w, activation='silu')
print(f"输出形状: {output.shape}")  # [2, 128, 11008]
```

---

## 🎯 示例 3：FP8 量化 GEMM

```python
import torch
from triton_ops import fp8_gemm, quantize_fp8

# 创建张量
a = torch.randn(1024, 4096, device='cuda', dtype=torch.float16)
b = torch.randn(4096, 2048, device='cuda', dtype=torch.float16)

# 方法 1：自动量化（推荐）
output = fp8_gemm(a, b)  # 自动量化并计算

# 方法 2：预量化输入
a_fp8, a_scale = quantize_fp8(a)
b_fp8, b_scale = quantize_fp8(b)
output = fp8_gemm(a_fp8, b_fp8, a_scale, b_scale)

print(f"输出形状: {output.shape}")  # [1024, 2048]
print(f"内存节省: 50%")
```

---

## 🏗️ 构建优化的 Transformer 层

```python
import torch
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP, FP8Linear

class OptimizedDecoderLayer(torch.nn.Module):
    """使用融合算子优化的 LLaMA 风格 Decoder"""
    
    def __init__(self, hidden_dim=4096, num_heads=32, intermediate_dim=11008):
        super().__init__()
        head_dim = hidden_dim // num_heads
        
        # 融合 RMSNorm + RoPE
        self.input_norm = FusedRMSNormRoPE(hidden_dim, head_dim)
        
        # 融合 Gated MLP
        self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim, activation='silu')
        
        # FP8 量化投影层
        self.q_proj = FP8Linear(hidden_dim, hidden_dim)
        self.k_proj = FP8Linear(hidden_dim, hidden_dim)
        self.v_proj = FP8Linear(hidden_dim, hidden_dim)
        self.o_proj = FP8Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, cos, sin):
        # Pre-norm 融合 RoPE
        normed = self.input_norm(x, cos, sin)
        
        # Attention 投影（FP8）
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)
        
        # Attention（使用 flash-attn 或自定义实现）
        # attn_out = flash_attn(q, k, v)
        
        # Output 投影 + 残差
        # x = x + self.o_proj(attn_out)
        
        # MLP + 残差
        x = x + self.mlp(x)
        return x

# 创建模型
layer = OptimizedDecoderLayer().cuda().half()
x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
cos = torch.randn(128, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(128, 64, device='cuda', dtype=torch.float16)

# 前向传播
output = layer(x, cos, sin)
print(f"输出形状: {output.shape}")
```

---

## 📊 验证性能

### 基准测试脚本

```python
import torch
import time
from triton_ops import fused_rmsnorm_rope

# 预热
torch.cuda.synchronize()

# 基准测试
x = torch.randn(8, 2048, 4096, device='cuda', dtype=torch.float16)
weight = torch.ones(4096, device='cuda', dtype=torch.float16)
cos = torch.randn(2048, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(2048, 64, device='cuda', dtype=torch.float16)

# 预热
for _ in range(10):
    _ = fused_rmsnorm_rope(x, weight, cos, sin)
torch.cuda.synchronize()

# 计时
start = time.perf_counter()
for _ in range(100):
    output = fused_rmsnorm_rope(x, weight, cos, sin)
torch.cuda.synchronize()
end = time.perf_counter()

avg_time = (end - start) / 100 * 1000
print(f"平均延迟: {avg_time:.3f} ms")
```

---

## 🎓 您学到了什么

- ✅ 如何使用函数式 API（`fused_rmsnorm_rope`、`fused_gated_mlp`、`fp8_gemm`）
- ✅ 如何使用 Module API（`FusedRMSNormRoPE`、`FusedGatedMLP`、`FP8Linear`）
- ✅ 如何构建优化的 Transformer 层
- ✅ 如何验证性能提升

---

## 🔗 下一步

- [示例教程](./examples.md) — 更多实用用例
- [集成指南](../guides/integration.md) — 与现有框架集成
- [API 参考](../api/kernels.md) — 完整 API 文档

---

<div align="center">

**[⬆ 返回顶部](#-快速开始指南)** | **[← 返回文档](../)**

</div>
