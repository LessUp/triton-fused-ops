---
title: 内部实现
---

# 内部实现

本节解释库的结构组织方式，以及这些 kernel 为何采用当前的实现方案。

<div class="link-grid link-grid-3">
  <a class="info-card" href="/zh/internals/architecture">
    <span class="card-kicker">架构</span>
    <strong>模块布局与职责划分</strong>
    <span>说明公开 API、验证层、kernel、自动调优和基准工具之间的关系。</span>
  </a>
  <a class="info-card" href="/zh/internals/kernel-design">
    <span class="card-kicker">Kernel</span>
    <strong>分块与融合策略</strong>
    <span>阅读 Triton kernel 的设计思路与内存访问模式。</span>
  </a>
  <a class="info-card" href="/zh/internals/memory-optimization">
    <span class="card-kicker">内存</span>
    <strong>HBM 访问削减与 SRAM 复用</strong>
    <span>理解为什么融合有效，以及项目如何减少全局显存往返。</span>
  </a>
</div>
