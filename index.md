---
layout: default
title: Knowledge Hub
nav_order: 1
head_title: "Triton Fused Ops Knowledge Hub"
description: "Bilingual technical knowledge hub for Triton fused kernels, FP8 quantization, autotuning, and benchmarking."
---

<div class="knowledge-home">
  <section class="hero-panel">
    <p class="section-eyebrow">Triton Fused Ops</p>
    <h1>知识首页 / Knowledge Hub</h1>
    <p class="hero-lead">
      面向 Transformer 推理路径的双语技术知识库，聚焦 Triton 融合算子、FP8 量化、自动调优、基准测试与实现细节。
      <br>
      A bilingual technical hub focused on Triton fused kernels, FP8 quantization, autotuning, benchmarking, and implementation details.
    </p>
    <div class="badge-row">
      <span class="metric-chip">RMSNorm + RoPE</span>
      <span class="metric-chip">Gated MLP</span>
      <span class="metric-chip">FP8 GEMM</span>
      <span class="metric-chip">Auto-Tuning</span>
      <span class="metric-chip">Benchmarking</span>
    </div>
  </section>

  <section>
    <h2>Language / 语言</h2>
    <div class="link-grid link-grid-2">
      <a class="info-card lang-card" href="{{ '/docs/en/' | relative_url }}">
        <span class="card-kicker">English</span>
        <strong>English Knowledge Base</strong>
        <span>API contracts, guides, benchmarks, and kernel internals.</span>
      </a>
      <a class="info-card lang-card" href="{{ '/docs/zh/' | relative_url }}">
        <span class="card-kicker">中文</span>
        <strong>中文知识库</strong>
        <span>接口说明、工程指南、性能知识与内部实现解析。</span>
      </a>
    </div>
  </section>

  <section>
    <h2>Knowledge Map / 知识地图</h2>
    <div class="link-grid link-grid-3">
      <div class="info-card">
        <span class="card-kicker">Start</span>
        <strong>Getting Started / 开始使用</strong>
        <span>安装、最小运行示例、模块封装用法。</span>
        <div class="card-links">
          <a href="{{ '/docs/en/getting-started/' | relative_url }}">EN</a>
          <a href="{{ '/docs/zh/getting-started/' | relative_url }}">中文</a>
        </div>
      </div>
      <div class="info-card">
        <span class="card-kicker">API</span>
        <strong>Kernels, FP8, Autotuner, Benchmark</strong>
        <span>公开接口、输入约束、数据模型与异常语义。</span>
        <div class="card-links">
          <a href="{{ '/docs/en/api/' | relative_url }}">EN</a>
          <a href="{{ '/docs/zh/api/' | relative_url }}">中文</a>
        </div>
      </div>
      <div class="info-card">
        <span class="card-kicker">Guide</span>
        <strong>Integration & Performance / 集成与性能</strong>
        <span>接入边界、性能测量、FP8 使用建议。</span>
        <div class="card-links">
          <a href="{{ '/docs/en/guides/' | relative_url }}">EN</a>
          <a href="{{ '/docs/zh/guides/' | relative_url }}">中文</a>
        </div>
      </div>
      <div class="info-card">
        <span class="card-kicker">Internals</span>
        <strong>Architecture / Kernel Design</strong>
        <span>模块组织、内存优化和 Triton kernel 设计思路。</span>
        <div class="card-links">
          <a href="{{ '/docs/en/internals/' | relative_url }}">EN</a>
          <a href="{{ '/docs/zh/internals/' | relative_url }}">中文</a>
        </div>
      </div>
      <div class="info-card">
        <span class="card-kicker">Scope</span>
        <strong>Runtime Boundaries / 运行边界</strong>
        <span>GPU 运行要求、CPU-safe 验证路径、连续内存与 dtype 约束。</span>
      </div>
      <div class="info-card">
        <span class="card-kicker">Reading Path</span>
        <strong>Recommended Study Order / 推荐阅读路径</strong>
        <span>Quick Start → API → Integration → Performance → Internals.</span>
      </div>
    </div>
  </section>

  <section>
    <h2>Kernel Focus / 核心知识点</h2>
    <div class="callout-grid">
      <div class="note-panel">
        <strong>`fused_rmsnorm_rope`</strong>
        <p>将 RMSNorm 与 RoPE 合并在同一条 kernel 路径中，减少中间 HBM 往返。</p>
      </div>
      <div class="note-panel">
        <strong>`fused_gated_mlp`</strong>
        <p>面向 SwiGLU/GeGLU 场景，将 gate、up 与激活计算合并。</p>
      </div>
      <div class="note-panel">
        <strong>`fp8_gemm`</strong>
        <p>提供 FP8 量化 GEMM 路径，支持自动量化输入与显式 scale 管理。</p>
      </div>
      <div class="note-panel">
        <strong>Supporting Knowledge</strong>
        <p>包含验证规则、异常模型、自动调优缓存、性能报告与源码级内部说明。</p>
      </div>
    </div>
  </section>
</div>
