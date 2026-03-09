---
layout: default
title: Triton Fused Ops
---

# Triton Fused Operators Library

高性能 Triton 算子库，专为 Transformer 模型优化，支持算子融合和 FP8 量化。

## 核心算子

| 算子 | 描述 | 关键优化 |
|------|------|----------|
| **RMSNorm + RoPE** | 归一化 + 旋转位置编码融合 | 3 次 HBM 访问 → 1 次 |
| **Gated MLP** | 门控 MLP 投影 + 激活融合 | 支持 SiLU / GELU |
| **FP8 GEMM** | 8-bit 浮点矩阵乘法 | 动态缩放，精度损失 < 1% |

## 技术特性

- **Auto-Tuning** — 自动搜索最优 BLOCK_SIZE、num_warps 等超参数
- **基准测试** — 对比 PyTorch/cuBLAS 基线，验证正确性和性能
- **函数式 + Module API** — 灵活的调用方式

## 快速开始

```bash
# 安装
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 运行基准测试
python benchmarks/benchmark_attention.py
python benchmarks/benchmark_gemm.py
```

## 使用示例

```python
import torch
from triton_ops import fused_rmsnorm_rope, fused_gated_mlp, fp8_gemm

# RMSNorm + RoPE 融合
x = torch.randn(2, 1024, 4096, device='cuda', dtype=torch.float16)
weight = torch.randn(4096, device='cuda', dtype=torch.float16)
cos = torch.randn(1024, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(1024, 64, device='cuda', dtype=torch.float16)
output = fused_rmsnorm_rope(x, weight, cos, sin)

# FP8 GEMM
a = torch.randn(1024, 4096, device='cuda', dtype=torch.float16)
b = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
output = fp8_gemm(a, b)
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.9+, Triton 2.1+ |
| 框架 | PyTorch 2.0+ |
| GPU | CUDA 11.8+ |
| 测试 | pytest |

## 链接

- [GitHub 仓库](https://github.com/LessUp/triton-fused-ops)
- [README](README.md)
