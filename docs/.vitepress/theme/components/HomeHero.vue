<template>
  <div class="home-hero">
    <div class="hero-content">
      <h1 class="hero-title">
        <span class="gradient">Triton Fused Ops</span>
      </h1>
      <p class="hero-tagline">{{ tagline }}</p>
      <p class="hero-abstract">{{ abstract }}</p>

      <!-- Metrics Strip -->
      <div class="metrics-strip">
        <div class="metric">
          <span class="value">~3×</span>
          <span class="label">{{ t.speedup }}</span>
        </div>
        <div class="metric">
          <span class="value">50%</span>
          <span class="label">{{ t.memory }}</span>
        </div>
        <div class="metric">
          <span class="value">4</span>
          <span class="label">{{ t.kernels }}</span>
        </div>
      </div>

      <!-- CTAs -->
      <div class="hero-actions">
        <a :href="links.getStarted" class="cta primary">{{ t.getStarted }}</a>
        <a :href="links.architecture" class="cta secondary">{{ t.architecture }}</a>
        <a href="https://github.com/LessUp/triton-fused-ops" class="cta outline">GitHub</a>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useData } from 'vitepress'

const { lang } = useData()

const i18n = {
  en: {
    tagline: 'High-Performance GPU Kernels for Transformer Inference',
    abstract: 'Production-ready Triton implementations of RMSNorm+RoPE fusion, Gated MLP fusion, and FP8 GEMM with auto-tuning infrastructure.',
    speedup: 'Speedup',
    memory: 'Memory ↓',
    kernels: 'Kernels',
    getStarted: 'Get Started',
    architecture: 'Architecture',
  },
  zh: {
    tagline: '面向 Transformer 推理的高性能 GPU 算子',
    abstract: '生产级 Triton 实现：RMSNorm+RoPE 融合、Gated MLP 融合、FP8 GEMM，配备自动调优基础设施。',
    speedup: '加速比',
    memory: '内存 ↓',
    kernels: '算子数',
    getStarted: '开始使用',
    architecture: '架构设计',
  },
}

const links = {
  en: {
    getStarted: '/en/getting-started/',
    architecture: '/en/internals/architecture',
  },
  zh: {
    getStarted: '/zh/getting-started/',
    architecture: '/zh/internals/architecture',
  },
}

const currentLang = computed(() => lang.value?.startsWith('zh') ? 'zh' : 'en')
const t = computed(() => i18n[currentLang.value])
const tagline = computed(() => t.value.tagline)
const abstract = computed(() => t.value.abstract)
</script>
