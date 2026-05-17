<template>
  <section class="whitepaper-hero" :aria-label="content.title">
    <div class="whitepaper-hero__body">
      <span class="editorial-kicker">{{ content.eyebrow }}</span>
      <h1 class="editorial-title">{{ content.title }}</h1>
      <p class="whitepaper-hero__lede">{{ content.dek }}</p>

      <div class="whitepaper-hero__highlights">
        <div
          v-for="(highlight, index) in content.highlights"
          :key="highlight.title"
          class="whitepaper-hero__highlight"
        >
          <span class="whitepaper-hero__highlight-index">{{ String(index + 1).padStart(2, '0') }}</span>
          <div>
            <strong>{{ highlight.title }}</strong>
            <p class="editorial-caption">{{ highlight.detail }}</p>
          </div>
        </div>
      </div>

      <div class="whitepaper-hero__actions">
        <a
          v-for="action in content.actions"
          :key="action.label"
          :href="action.href"
          class="whitepaper-hero__action"
          :class="{ 'whitepaper-hero__action--primary': action.primary }"
        >
          <span>{{ action.label }}</span>
          <span aria-hidden="true">→</span>
        </a>
      </div>
    </div>

    <aside class="whitepaper-hero__aside">
      <div class="whitepaper-hero__sheet">
        <div
          v-for="fact in content.facts"
          :key="fact.label"
          class="whitepaper-hero__sheet-row"
        >
          <span class="whitepaper-hero__sheet-label">{{ fact.label }}</span>
          <span class="whitepaper-hero__sheet-value">{{ fact.value }}</span>
        </div>
      </div>

      <div class="whitepaper-hero__rail">
        <div
          v-for="panel in content.rail"
          :key="panel.title"
          class="whitepaper-hero__rail-item"
        >
          <strong>{{ panel.title }}</strong>
          <p class="editorial-caption">{{ panel.detail }}</p>
        </div>
      </div>
    </aside>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'

interface HeroAction {
  href: string
  label: string
  primary?: boolean
  external?: boolean
}

interface HeroHighlight {
  title: string
  detail: string
}

interface HeroFact {
  label: string
  value: string
}

interface HeroRailItem {
  title: string
  detail: string
}

interface HeroContent {
  eyebrow: string
  title: string
  dek: string
  actions: HeroAction[]
  highlights: HeroHighlight[]
  facts: HeroFact[]
  rail: HeroRailItem[]
}

const { lang } = useData()

const localeContent: Record<'en' | 'zh', HeroContent> = {
  en: {
    eyebrow: 'Engineering Field Guide',
    title: 'A homepage shaped like technical review, not a metrics strip',
    dek: 'Trace Triton Fused Ops through kernel families, validation seams, and benchmark evidence before you ever drop into the implementation details.',
    actions: [
      { label: 'Enter the academy', href: '/academy/', primary: true },
      { label: 'Inspect the architecture lab', href: '/architecture-lab/' },
      { label: 'Browse kernel families', href: '/kernel-families/' },
      { label: 'GitHub', href: 'https://github.com/LessUp/triton-fused-ops', external: true },
    ],
    highlights: [
      {
        title: 'Lead with operating context',
        detail: 'The first screen explains what the kernel library covers, how to evaluate it, and where each reader should go next.',
      },
      {
        title: 'Editorial structure over launch-page theatrics',
        detail: 'Highlights, facts, and reading rails follow the same whitepaper system introduced across the Task 2 docs rebuild.',
      },
      {
        title: 'Built for code-adjacent reading',
        detail: 'Sections are optimized for engineers who compare docs against source, validation contracts, and benchmark methodology.',
      },
    ],
    facts: [
      { label: 'Kernel families', value: 'RMSNorm + RoPE, Gated MLP, FP8 GEMM' },
      { label: 'Validation mode', value: 'CPU references plus runtime contracts' },
      { label: 'Reading paths', value: 'Installation, architecture, academy' },
      { label: 'Voice', value: 'Concise, technical, evidence-backed' },
    ],
    rail: [
      {
        title: 'For evaluators',
        detail: 'Jump from the hero into architecture and academy notes without passing through a marketing funnel.',
      },
      {
        title: 'For implementers',
        detail: 'Installation and downstream sections stay one click away, with locale-aware internal routes for deployed docs bases.',
      },
    ],
  },
  zh: {
    eyebrow: '工程导读',
    title: '首页首先呈现技术评审视角，而不是指标条幅',
    dek: '先从算子家族、验证边界与基准证据理解 Triton Fused Ops，再进入更细的实现细节。',
    actions: [
      { label: '进入学院导读', href: '/academy/', primary: true },
      { label: '查看架构实验室', href: '/architecture-lab/' },
      { label: '浏览算子族', href: '/kernel-families/' },
      { label: 'GitHub', href: 'https://github.com/LessUp/triton-fused-ops', external: true },
    ],
    highlights: [
      {
        title: '先交代运行语境',
        detail: '首屏说明库覆盖的算子范围、评估方式，以及不同读者下一步该去哪里。',
      },
      {
        title: '延续白皮书式编排',
        detail: '高亮、事实卡与阅读侧栏沿用 Task 2 文档重建引入的 editorial 系统，而不是回到营销页套路。',
      },
      {
        title: '适合对照代码阅读',
        detail: '整体节奏服务于需要同时查看源码、校验约束与基准方法的工程读者。',
      },
    ],
    facts: [
      { label: '算子家族', value: 'RMSNorm + RoPE、Gated MLP、FP8 GEMM' },
      { label: '验证方式', value: 'CPU 参考实现 + 运行时输入校验' },
      { label: '阅读路径', value: '安装、架构、学院导读' },
      { label: '语气', value: '克制、技术化、强调证据' },
    ],
    rail: [
      {
        title: '给评审者',
        detail: '从首页可直接进入架构和学院说明，不需要穿过营销化叙事漏斗。',
      },
      {
        title: '给实现者',
        detail: '安装与后续章节始终保持一步可达，并通过 base-aware 链接适配部署后的文档路径。',
      },
    ],
  },
}

const localeKey = computed<'en' | 'zh'>(() => (lang.value?.startsWith('zh') ? 'zh' : 'en'))
const localePrefix = computed(() => `/${localeKey.value}`)

function toLocalizedHref(path: string): string {
  return withBase(`${localePrefix.value}${path}`)
}

const content = computed<HeroContent>(() => {
  const fallback = localeContent[localeKey.value]

  return {
    ...fallback,
    actions: fallback.actions.map((action) => ({
      ...action,
      href: action.external ? action.href : toLocalizedHref(action.href),
    })),
  }
})
</script>
