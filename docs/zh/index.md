---
layout: page
title: Triton Fused Ops
description: 面向 Transformer 推理的高性能 Triton 算子库
---

<script setup>
import HomeHero from '@theme/components/HomeHero.vue'
import KernelShowcase from '@theme/components/KernelShowcase.vue'
import ArchitecturePreview from '@theme/components/ArchitecturePreview.vue'
</script>

<HomeHero />

## 核心算子

<KernelShowcase />

## 架构概览

<ArchitecturePreview />

## 为什么选择 Triton Fused Ops？

| 特性 | 优势 |
|------|------|
| **算子融合** | RMSNorm + RoPE 单次 kernel launch，消除中间结果的 HBM 往返 |
| **CPU 参考实现** | 纯 NumPy 实现，无需 GPU 即可验证正确性 |
| **自动调优** | TritonAutoTuner 自动发现针对硬件的最优启动参数 |
| **FP8 支持** | E4M3/E5M2 量化 GEMM，权重内存减少 50% |

## 快速开始

```bash
pip install triton-fused-ops
```

```python
import torch
from triton_ops import fused_rmsnorm_rope

x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
weight = torch.ones(4096, device="cuda", dtype=torch.float16)
cos = torch.randn(128, 64, device="cuda", dtype=torch.float16)
sin = torch.randn(128, 64, device="cuda", dtype=torch.float16)

y = fused_rmsnorm_rope(x, weight, cos, sin)
```

## 下一步

- [安装指南](/zh/getting-started/installation) — 详细安装说明
- [API 参考](/zh/api/kernels) — 算子文档
- [架构设计](/zh/internals/architecture) — 内部实现原理
- [性能优化](/zh/guides/performance) — 优化技巧
