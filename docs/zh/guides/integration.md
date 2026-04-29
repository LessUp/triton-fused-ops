---
layout: default
title: 集成指南
parent: 工程指南
grand_parent: 中文文档
nav_order: 1
description: "如何把仓库当前导出的 kernel 接入更大的模型代码"
---

# 集成指南

本页聚焦“当前仓库真实存在的代码”应该怎样接入，而不是把仓库描述成完整框架适配层。

## 先选集成层级

### 函数式 API

适合场景：

- 你的模型自己持有权重和 cache，
- 你只想替换少数热点路径，
- 你希望控制更细粒度的调用边界。

典型函数：

- `fused_rmsnorm_rope`
- `fused_gated_mlp`
- `fp8_gemm`
- `quantize_fp8` / `dequantize_fp8`

### 模块封装

适合场景：

- 你希望用 `nn.Module` 方式组合，
- 你希望权重由模块内部持有，
- 你在搭建推理导向的 block 级结构。

典型模块：

- `FusedRMSNormRoPE`
- `FusedGatedMLP`
- `FP8Linear`

## 关键运行边界

### `FusedRMSNormRoPE` 不是普通 norm 层

它的前向签名需要显式传入 RoPE 输入：

```python
out = module(x, cos, sin)
```

所以它更适合放在你已经持有位置 cache 的模型边界里。

### `FusedGatedMLP` 只覆盖 gated expansion

仓库里的模块返回的是 gated 中间输出。完整 decoder FFN 还需要在外部补上 down projection 和 residual 路径。

### `FP8Linear` 更适合推理型场景

该模块第一次前向时会量化并缓存权重。如果之后浮点权重继续更新，缓存的 FP8 权重并不会自动刷新。

## Decoder block 草图

```python
import torch
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP, FP8Linear

class DecoderSlice(torch.nn.Module):
    def __init__(self, hidden_dim=4096, num_heads=32, intermediate_dim=11008):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.norm = FusedRMSNormRoPE(hidden_dim, head_dim)
        self.q_proj = FP8Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = FP8Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = FP8Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim, activation="silu")

    def forward(self, x, cos, sin):
        normed = self.norm(x, cos, sin)
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)
        mlp_out = self.mlp(normed)
        return q, k, v, mlp_out
```

## HuggingFace / 自定义模型 patch 的现实做法

仓库本身没有提供官方的 HuggingFace 或 vLLM 适配器。更实际的接入模式是：

1. 找到模型里真正持有这些张量的子模块。
2. 只替换 norm / projection / MLP 这些热点片段。
3. 保留模型原有的 attention 实现，除非你同时掌控那部分路径。
4. 用代表性输入把 patch 前后数值对齐验证一遍。

换句话说，应把本仓库当作“优化原语集合”，而不是“现成框架插件”。

## 接入前检查清单

- 输入必须在 CUDA 上，
- 输入必须 contiguous，
- RoPE cache 形状要确认清楚，
- `hidden_dim` 必须与 `head_dim` 能正确整除，
- benchmark 时要加 warmup 和同步，
- rollout 前必须和未融合 baseline 对齐输出。
