---
title: 参考文献
description: Triton Fused Ops 背后的核心论文与技术参考资料
---

# 参考文献

下面这些材料入选，不是因为它们全面，而是因为它们分别解释了某个 Kernel family 的数学内容，或者解释了本仓库用来讨论性能与系统设计的语言。

## 核心参考

| 文献 | 为什么与本仓库直接相关 |
| :-- | :-- |
| FlashAttention | 提供 IO-aware 的分析框架，让“为什么融合值得做”不再停留在口号层。 |
| FP8 Formats for Deep Learning | 让 FP8 栈建立在真实格式讨论上，而不是模糊的 8-bit 叙事。 |
| RoFormer | 解释 RMSNorm + RoPE family 中 rotary embedding 的数学定义。 |
| Root Mean Square Layer Normalization | 定义 attention 准备阶段所使用的规范化变体。 |

## 扩展阅读

- **Tillet, Kung, Cox — Triton**：这些 kernel 能以 Python 表达出来，靠的是 Triton 的编译器模型。
- **Dao 等 — FlashAttention / FlashAttention-2**：为重 fusion 的 GPU 代码提供 IO-aware 与 work partitioning 的经验。
- **Micikevicius 等 — FP8 Formats for Deep Learning**：解释 E4M3 与相关 FP8 语义。
- **Dettmers 等 — LLM.int8()**：理解低精度部署取舍与误差处理时的有用背景。
- **Xiao 等 — SmoothQuant**：当你考虑 scale 移动与量化工程性时值得参考。
- **Su 等 — RoFormer**：rotary embedding 的形式化来源。
- **Zhang 与 Sennrich — Root Mean Square Layer Normalization**：RMSNorm 的正式定义。
- **Williams、Waterman、Patterson — Roofline**：讨论带宽与吞吐取舍时很紧凑的性能解释工具。

## 阅读启发式

论文负责回答“为什么会有这个 family”，而仓库代码负责回答“这个想法在这里是如何被操作化的”。两者不能互相替代。
