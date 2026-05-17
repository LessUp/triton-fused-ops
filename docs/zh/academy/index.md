---
title: 学院
description: 以白皮书叙事方式理解 Triton Fused Ops 的学习路径
---

# 学院

学院是整套文档里的叙事主路径。它不把 API 孤立罗列，而是解释系统如何组合、每一层为什么存在，以及读者该按什么顺序建立心智模型。

## 学院地图

| 顺序 | 路线 | 为什么放在这里 |
| :-- | :-- | :-- |
| 1 | [系统总览](./system-overview) | 先把库理解成分层系统，再看局部实现 |
| 2 | [算子族](../kernel-families/) | 用工作负载、契约与证据语言理解每个面向用户的操作 |
| 3 | [架构实验室](../architecture-lab/) | 检查模块缝隙、公共导出与运行时契约 |
| 4 | [工程指南](../guides/) | 从理解转向使用、测量与集成 |
| 5 | [参考与研究](../reference-research/) | 把仓库放进更大的推理系统与 kernel 生态里看 |

规范路由：`/zh/academy/system-overview`、`/zh/kernel-families/`、`/zh/architecture-lab/`。

## 三种阅读方式

### 给评估者

先读[系统总览](./system-overview)，再跳到[运行时契约](../architecture-lab/runtime-contracts)。这条路径适合审查实现是否克制、边界是否清晰。

### 给 kernel 工程师

接着读[算子族](../kernel-families/)，再对照 `triton_ops.kernels` 与 `triton_ops.reference`。这条路径的重点是看清 fusion、reference 数学与 validation 在哪里汇合。

### 给性能工程实践者

先看系统总览，再直接进入[性能指南](../guides/performance)和[参考与研究](../reference-research/)。这条路径关心 Benchmarking、Auto-Tuning 与 Performance metrics 应该怎样分层解读。

## 学院强调什么

- kernel family 才是用户侧最稳定的推理单元；
- Benchmarking 是证据整理，不是展示装饰；
- Auto-Tuning 是边界明确、只对 latency 负责的子系统；
- Performance metrics 只有在问题形状明确时才成立。
