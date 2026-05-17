---
title: Fused RMSNorm + RoPE
description: Triton Fused Ops 中把 RMSNorm 与旋转位置编码合并的算子族
---

# Fused RMSNorm + RoPE

`fused_rmsnorm_rope` 是仓库里最直观的融合故事：规范化一次、旋转一次，并且避免把中间规范化结果显式写回 HBM。

## 这个 family 做了什么

它把 Transformer attention 准备阶段里常见的两步组合到一起：

1. 基于学习到的权重向量执行 **RMSNorm**；
2. 使用预计算 `cos` / `sin` 执行 **RoPE**。

用户可见的接口包括：

- `fused_rmsnorm_rope`
- `FusedRMSNormRoPE`
- `reference_fused_rmsnorm_rope`（用于正确性比对）

## 为什么它是独立的 Kernel family

它的主张很集中：把规范化结果留在即将消费它的计算附近，从而减少内存流量。

因此评价这个 family 时，重点应放在“内存流量是否明显下降”，而不是把它抽象成笼统的“融合越多越好”。

## 契约表面

这个 family 的运行时边界比普通 norm 层更具体：

| 输入 | 角色 |
| :-- | :-- |
| `x` | `[batch, seq_len, hidden_dim]` 激活输入 |
| `weight` | `[hidden_dim]` 形状的 RMSNorm 权重 |
| `cos`, `sin` | 与序列长度、head 维度匹配的 RoPE 表 |
| `num_heads` | 当你不希望自动推断时可显式传入 |

当前 launcher 会验证 CUDA 放置、支持的浮点 dtype、contiguous 布局、`x` / `weight` / `cos` / `sin` 的 shape 一致性、正的 `eps`、偶数 `head_dim`，以及 batch / sequence / hidden 维度为正。若省略 `num_heads`，只有当 `hidden_dim` 能被 RoPE 的 head 维整除时才会自动推断。

## 验证与证据

相关的证据面比较直接：

- launch 期验证主要落在 `validate_rmsnorm_rope_inputs` 与 `validate_eps`、`validate_head_dim`、`validate_positive_dimensions`；
- reference 实现提供可直接阅读的数学基线；
- Benchmarking 应将 fused 路径与明确的 unfused baseline 对照。

## Benchmarking 指南

Benchmarking 在这里要回答的是一个窄问题：对真实的 sequence length 与 hidden size 来说，这次融合减少的流量是否足以产生可辩护的收益？

因此性能审阅应记录：

- warmup 与同步计时；
- 代表性的 `seq_len` 与 `hidden_dim`；
- 与 reference 路径的正确性比对；
- 在合适场景下强调内存敏感行为的 Performance metrics。

## 集成提示

`FusedRMSNormRoPE` 最适合放在调用方已经持有 RoPE cache 的边界上。它不是每个 norm 节点的通用替换件，而是 attention pipeline 里规范化与 rotary embedding 相邻的那一段。
