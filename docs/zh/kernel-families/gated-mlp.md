---
title: Fused Gated MLP
description: 面向 SwiGLU 与 GeGLU 风格前馈扩展阶段的融合算子族
---

# Fused Gated MLP

`fused_gated_mlp` 面向现代 Transformer block 中的 gated feed-forward expansion 阶段。它把两次投影、一个 activation 和一次逐元素 gate 合进同一次 launch。

## family 概览

代码同时支持 **SwiGLU**（`silu`）与 **GeGLU**（`gelu`）两种 activation 选择。用户可见 API 包括：

- `fused_gated_mlp`
- `FusedGatedMLP`
- `reference_gated_mlp`

## 数据契约

这个 family 接收：

- 形状为 `[batch, seq_len, hidden_dim]` 的 `x`；
- 形状为 `[intermediate_dim, hidden_dim]` 的 `gate_weight`；
- 形状为 `[intermediate_dim, hidden_dim]` 的 `up_weight`；
- 决定 `silu` 或 `gelu` 的 `activation` 字符串。

这里最关键的设计变量是 `intermediate_dim`，因为它直接决定扩展宽度，也就决定了算术量与内存足迹。

## 实际融合了什么

实现会从同一块输入 tile 计算两次矩阵乘，应用选定 activation，再把两路结果相乘得到 gated activation 输出。

这个范围比完整前馈网络窄：down projection、residual 处理与 block 级编排仍然在这个 Kernel family 之外。

## 审查时该关注什么

1. activation 选择是否与目标模型一致；
2. `intermediate_dim` 是否真实反映模型宽度；
3. 正确性检查是否确实对照 reference 实现；
4. latency 是否放回完整 FFN 路径里解释，而不是只盯着融合子步骤。

## Benchmarking 立场

这里的 Benchmarking 不只是单点理论瓶颈问题，更关心重复 launch 与中间结果搬运的实际成本。实验应按 activation 拆分，并且对照一个真正模拟待替换模型代码的 unfused baseline。
