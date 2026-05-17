---
title: 演进思路
description: 一个工业化、evidence-backed 的 kernel 库应如何继续演进而不失焦
---

# 演进思路

这个仓库合理的演进方向应当是工业化的，而不是无边界扩张。一个成熟 kernel 库赢得信任的方式，是持续保持 evidence-backed、持续收紧证明面，并且谨慎选择下一个问题。

## 下一阶段的设计原则

### 把 kernel family 保持为增长单位

只有当仓库能像解释现有 family 那样，清楚解释新 family 的契约、reference 路径与性能评估方式时，新的 kernel family 才值得落地。

### 优先 evidence-backed 的增量

页面更多、辅助函数更多、kernel 更多，本身都不等于进步。真正的标准是：评审者能否不靠猜测地验证语义、运行时契约与测量方法？

### 把工业化取舍写明白

工业化库不应隐藏自己的边界。它应明确说明某个 wrapper 是否面向推理、某项改动上线前是否必须做 Benchmarking，以及在什么场景下 Auto-Tuning 或 Performance metrics 根本不是合适语言。

## 值得继续追问的下一个问题

- 哪些公共表面还可以更简单，而又不隐藏重要控制权？
- 哪些证明面仍然过于隐式？
- benchmark 方法学、reference 覆盖或部署适配器，下一步最值得回答的是什么？

这一页的价值不在预测未来，而在于让未来演进继续贴住仓库最强的特性：对每个系统决策都给出紧凑、evidence-backed 的说明。
