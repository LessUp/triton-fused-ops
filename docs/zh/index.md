---
layout: default
title: 中文文档
nav_title: 中文
nav_order: 3
has_children: true
permalink: /docs/zh/
---

# 中文知识库

这里聚合了与当前仓库实现一致的技术内容：公开 API、运行约束、性能工具链，以及 Triton kernel 的内部设计说明。

<div class="link-grid link-grid-2">
  <a class="info-card" href="{{ '/docs/zh/getting-started/' | relative_url }}">
    <span class="card-kicker">开始使用</span>
    <strong>安装、运行与最小示例</strong>
    <span>从环境准备、第一段可运行代码，到模块封装示例。</span>
  </a>
  <a class="info-card" href="{{ '/docs/zh/api/' | relative_url }}">
    <span class="card-kicker">API 参考</span>
    <strong>公开接口与输入契约</strong>
    <span>涵盖 kernel、量化、自动调优、基准工具、数据模型与异常说明。</span>
  </a>
  <a class="info-card" href="{{ '/docs/zh/guides/' | relative_url }}">
    <span class="card-kicker">工程指南</span>
    <strong>集成与性能知识</strong>
    <span>说明如何接入融合算子、如何正确测量性能、如何使用 FP8。</span>
  </a>
  <a class="info-card" href="{{ '/docs/zh/internals/' | relative_url }}">
    <span class="card-kicker">内部实现</span>
    <strong>源码级实现背景</strong>
    <span>查看架构分层、kernel 设计取舍与内存访问优化思路。</span>
  </a>
</div>

## 推荐阅读路径

<div class="callout-grid">
  <div class="note-panel">
    <strong>第一次阅读</strong>
    <p>先看 <a href="{{ '/docs/zh/getting-started/installation/' | relative_url }}">安装指南</a> 与 <a href="{{ '/docs/zh/getting-started/quickstart/' | relative_url }}">快速开始</a>。</p>
  </div>
  <div class="note-panel">
    <strong>准备接入项目</strong>
    <p>先看 <a href="{{ '/docs/zh/api/kernels/' | relative_url }}">核心算子</a> 与 <a href="{{ '/docs/zh/guides/integration/' | relative_url }}">集成指南</a>。</p>
  </div>
  <div class="note-panel">
    <strong>准备做性能工作</strong>
    <p>先看 <a href="{{ '/docs/zh/api/benchmark/' | relative_url }}">基准测试</a>、<a href="{{ '/docs/zh/api/autotuner/' | relative_url }}">自动调优</a>、<a href="{{ '/docs/zh/guides/performance/' | relative_url }}">性能优化</a>。</p>
  </div>
  <div class="note-panel">
    <strong>准备读源码</strong>
    <p>先看 <a href="{{ '/docs/zh/internals/architecture/' | relative_url }}">架构设计</a> 与 <a href="{{ '/docs/zh/internals/kernel-design/' | relative_url }}">算子设计</a>。</p>
  </div>
</div>

## 运行边界提醒

- Triton kernel 的实际执行需要 CUDA。
- CPU-only 环境仍适合导入检查、lint、类型检查、构建，以及 CPU-safe 测试。
- 站点现在只保留技术知识页，不再把更新日志和仓库流程信息发布到 GitHub Pages。
