---
title: 性能指南
description: 如何在 Triton Fused Ops 中正确做 Benchmarking、Auto-Tuning 与性能解释
---

# 性能指南

在这个仓库里讨论性能，前提是始终把三个词分开：**Benchmarking**、**Auto-Tuning**、**Performance metrics**。

## 三层职责

| 关注点 | 主要工具 | 它回答什么问题 |
| :-- | :-- | :-- |
| Benchmarking | `BenchmarkSuite`、`CorrectnessVerifier`、`PerformanceReport` | 某个 Kernel family 在声明的方法下测得多快，并且是否仍然正确？ |
| Auto-Tuning | `TritonAutoTuner`、`ConfigCache`、预设配置空间 | 哪组 launch 参数把 latency 压到最低？ |
| Performance metrics | `PerformanceProfile`、`measure_latency`、`measure_metrics`、`compute_metrics` | 在已知 shape 上下文下，这个 latency 意味着怎样的吞吐或带宽，以及这个 latency 是如何测出来的？ |

## 先做 Benchmarking

先做 Benchmarking，因为它回答最基础的问题：测到的 kernel 既快不快，也对不对。

严肃的 Benchmarking 至少应包含：

- warmup 轮次；
- 计时区间前后的显式 `torch.cuda.synchronize()`；
- 来自真实模型的代表性 shape；
- 与 reference 实现的正确性比对。

`BenchmarkSuite` 已经把这些流程封装在一起，并且把实际计时循环委托给 `triton_ops.performance.measure_metrics`，这样同步纪律只需要维护一处。

## Auto-Tuning 放在哪

Auto-Tuning 用来搜索 block size、warp 数等 launch 参数，不应与端到端 benchmark 混为一谈。

一个实用规则是：

- 当你拥有 callable 并想找最佳配置时，用 **Auto-Tuning**；
- 当你需要把结果提交为证据时，用 **Benchmarking**。

## 如何解释 Performance metrics

单独的 latency 不是完整故事。只有在问题 shape 明确后，`triton_ops.performance` 的 Performance metrics 才有意义。

- elementwise / reduction 风格路径通常更适合看有效带宽；
- GEMM 风格路径通常更适合看吞吐；
- 没有 shape 上下文时，结论应停留在 latency。

## 各 family 的提醒

### RMSNorm + RoPE

把它视为内存敏感型 family。Benchmarking 应覆盖有代表性的 sequence length 与 hidden size。

### Gated MLP

把它视为 launch 次数与融合效率的故事。实验记录里必须写清 activation 选择。

### FP8 栈

把它视为格式、scale 与矩阵乘行为共同决定的路径。需要记录输入是预先量化还是运行时自动量化。

## 不该做什么

- 不要把 Auto-Tuning 结果包装成完整 Benchmarking 证据；
- 不要在没有说明 shape 假设时报告 Performance metrics；
- 不要把单个 kernel 的收益直接外推成整模型提速。
