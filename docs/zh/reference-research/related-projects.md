---
title: 相关项目
description: 与 Triton Fused Ops 相邻的开源项目及其比较价值
---

# 相关项目

这些项目之所以重要，不是因为它们有名，而是因为它们定义了 kernel 编写、tensor runtime 行为与推理部署的生态预期。下面的评论刻意保持克制：每个项目究竟帮我们比较什么？

## 核心基础设施

| 项目 | 为什么这里要看它 |
| :-- | :-- |
| OpenAI Triton | 本仓库所有 Triton kernel 的编译器与 Python DSL 基础。它不只是构建依赖，而是实现媒介本身。 |
| PyTorch | 公共 API、模块封装与大部分 validation 假设所依赖的 tensor runtime。它决定了用户契约的基底。 |

## 推理与系统侧邻居

| 项目 | 经过筛选的比较视角 |
| :-- | :-- |
| vLLM | 适合作为部署语境参照。本仓库目前不提供 vLLM 适配器，但 vLLM 能帮助我们理解真正的 serving 系统对优化原语有什么期待。 |
| TensorRT-LLM | 适合作为工业化推理栈对比项，尤其是在讨论“一个聚焦的 kernel 库之外，生产栈还包含哪些部件”时。 |
| xFormers | 有助于理解另一个项目如何打包高效 Transformer building blocks，而又不声称替代完整框架栈。 |
| CUTLASS | 适合在思考矩阵乘结构、tile ordering，以及更低层 kernel 模式如何影响 FP8 GEMM 设计时对照。 |

## 实用阅读规则

相关项目的价值在于把问题磨尖，而不是借它们给仓库贴金。真正有用的问题从来不是“文档有没有提到大项目”，而是“哪个相邻项目能澄清这一个设计决策的取舍”。
