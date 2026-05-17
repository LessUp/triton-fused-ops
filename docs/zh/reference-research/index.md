---
title: 参考与研究
description: Triton Fused Ops 的相关项目、参考文献与技术演进思考
---

# 参考与研究

这一节是整套文档里偏研究视角的一半。它的目的不是靠列名词抬高项目，而是给出一个研究议程：怎样把本仓库与相邻系统工作做有约束的比较。

## 研究议程

一个有效的研究议程至少包含三部分：

1. **相关项目** —— 周边系统解决了什么问题，它们对部署预期意味着什么；
2. **参考文献** —— 哪些论文解释了数学基础、编译器模型与性能话语；
3. **演进思路** —— 一个工业化、evidence-backed 的 kernel 库应该如何继续收敛式演进。

<div class="link-grid link-grid-3">
  <a class="info-card" href="./related-projects">
    <span class="card-kicker">项目</span>
    <strong>相邻系统的设计参照</strong>
    <span>从 API 形态、部署姿态与 kernel 范围三个维度比较周边项目。</span>
  </a>
  <a class="info-card" href="./references">
    <span class="card-kicker">文献</span>
    <strong>经过筛选的论文与技术资料</strong>
    <span>回到 Triton、FlashAttention、FP8、RoPE 与 RMSNorm 的源头材料。</span>
  </a>
  <a class="info-card" href="./evolution-thinking">
    <span class="card-kicker">演进</span>
    <strong>如何谨慎地把仓库做深</strong>
    <span>用演进笔记讨论范围、证明面以及真正值得回答的下一批问题。</span>
  </a>
</div>
