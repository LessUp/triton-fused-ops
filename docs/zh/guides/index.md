---
layout: default
title: 工程指南
parent: 中文文档
nav_order: 3
has_children: true
permalink: /docs/zh/guides/
---

# 工程指南

这些页面聚焦工程决策：融合算子应放在模型的哪个边界、性能该如何正确测量、FP8 的收益和风险如何取舍。

<div class="link-grid link-grid-3">
  <a class="info-card" href="{{ '/docs/zh/guides/integration/' | relative_url }}">
    <span class="card-kicker">集成</span>
    <strong>运行契约与模型边界</strong>
    <span>帮助你在函数式 API、模块封装与自定义适配器之间做选择。</span>
  </a>
  <a class="info-card" href="{{ '/docs/zh/guides/performance/' | relative_url }}">
    <span class="card-kicker">性能</span>
    <strong>测量方法与调优路径</strong>
    <span>说明如何做正确基准测试、如何理解指标、如何调优自定义 kernel。</span>
  </a>
  <a class="info-card" href="{{ '/docs/zh/guides/fp8-best-practices/' | relative_url }}">
    <span class="card-kicker">FP8</span>
    <strong>量化最佳实践</strong>
    <span>说明 FP8 适合的位置，以及数值敏感步骤应保留更高精度。</span>
  </a>
</div>
