---
layout: page
title: Triton Fused Ops
description: Triton Fused Ops 中文白皮书首页
---

<script setup>
import WhitepaperHero from '@theme/components/WhitepaperHero.vue'
import ReaderTracks from '@theme/components/ReaderTracks.vue'
import KernelAtlas from '@theme/components/KernelAtlas.vue'
import SystemBlueprint from '@theme/components/SystemBlueprint.vue'
import ResearchLandscape from '@theme/components/ResearchLandscape.vue'
</script>

<WhitepaperHero />
<ReaderTracks />
<KernelAtlas />
<SystemBlueprint />

## 这个仓库实际交付什么

Triton Fused Ops 是一个面向 Transformer 推理的聚焦型 GPU kernel 库，不试图充当完整模型框架。仓库交付的是少量可直接部署的 **Kernel family**，以及围绕它们构建的 reference 层、validation 层、Benchmarking、Auto-Tuning 与 Performance metrics 工具链。

<div class="link-grid link-grid-3">
  <a class="info-card" href="./overview/">
    <span class="card-kicker">导读</span>
    <strong>先统一术语，再看实现</strong>
    <span>从项目词汇、证据模型与阅读顺序入手，避免把性能、调优和正确性混为一谈。</span>
  </a>
  <a class="info-card" href="./academy/">
    <span class="card-kicker">学院</span>
    <strong>按系统层次读完整故事</strong>
    <span>先看系统总览，再下钻到算子族与架构边界。</span>
  </a>
  <a class="info-card" href="./guides/">
    <span class="card-kicker">工程指南</span>
    <strong>把结论接进真实代码</strong>
    <span>在做集成与性能决策时，直接使用工程指南而不是凭印象推断。</span>
  </a>
</div>

## 如何审查这个仓库里的技术主张

| 问题 | 去哪里看 | 什么算证据 |
| :-- | :-- | :-- |
| 对外承诺是什么？ | `triton_ops.__init__`、API 页面、算子族页面 | 导出的 launcher、模块封装、辅助函数与它们的契约 |
| 正确性如何核对？ | `triton_ops.reference`、`triton_ops.validation`、`BenchmarkSuite` | reference 实现、显式验证、correctness 校验 |
| 延迟结论怎么表达？ | Benchmarking 文档、benchmark 套件、performance 工具 | warmup、同步、明确问题形状、派生指标 |
| 调优边界停在哪里？ | `triton_ops.autotuner`、性能指南 | Auto-Tuning 只搜索低延迟配置，不会偷偷改写运行时语义 |

## 面向不同角色的阅读顺序

1. **评估者**：先读[导读](./overview/)、[学院](./academy/)、[架构实验室](./architecture-lab/)。
2. **集成者**：从[算子族](./kernel-families/)进入，再读[集成指南](./guides/integration)。
3. **性能审阅者**：先看[性能指南](./guides/performance)，再看[参考与研究](./reference-research/)。

<ResearchLandscape />
