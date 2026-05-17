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

const props = defineProps<{
  eyebrow?: string
  title?: string
  dek?: string
  actions?: HeroAction[]
  highlights?: HeroHighlight[]
  facts?: HeroFact[]
  rail?: HeroRailItem[]
}>()

const { lang } = useData()

const localeContent: Record<'en' | 'zh', HeroContent> = {
  en: {
    eyebrow: 'Technical Whitepaper',
    title: 'Editorial framing for Triton fused operators',
    dek: 'A premium docs surface for engineers who evaluate kernel craft, validation boundaries, and system design under real production scrutiny.',
    actions: [
      { label: 'Read the architecture lab', href: '/en/architecture-lab/', primary: true },
      { label: 'Enter the academy', href: '/en/academy/' },
    ],
    highlights: [
      {
        title: 'Research pacing over launch theatrics',
        detail: 'Lead with operating assumptions, evidence paths, and the architecture seams reviewers actually inspect.',
      },
      {
        title: 'Dual-theme ready without palette collapse',
        detail: 'Tokens preserve legibility on bright daytime panels and darker night reading sessions without gimmicks.',
      },
      {
        title: 'Reusable figures for later sections',
        detail: 'Shared components establish consistent hero, blueprint, atlas, and research framing for every downstream page.',
      },
    ],
    facts: [
      { label: 'Edition', value: 'Docs surface rebuild / task 2' },
      { label: 'Audience', value: 'Senior engineers and interview reviewers' },
      { label: 'Voice', value: 'Technical, concise, evidence-backed' },
      { label: 'Coverage', value: 'Kernels, validation, benchmarking, research' },
    ],
    rail: [
      {
        title: 'Reading mode',
        detail: 'Strong left alignment and dense captions keep scanning fast when comparing docs beside code.',
      },
      {
        title: 'Figure discipline',
        detail: 'Frames, notes, and section rhythm behave like a whitepaper rather than a marketing landing page.',
      },
    ],
  },
  zh: {
    eyebrow: '技术白皮书',
    title: '为 Triton 融合算子建立工程化叙事',
    dek: '面向严苛评审者的高质量文档界面，突出 kernel 工艺、验证边界与系统设计，而不是营销化装饰。',
    actions: [
      { label: '阅读架构实验室', href: '/zh/architecture-lab/', primary: true },
      { label: '进入学院导读', href: '/zh/academy/' },
    ],
    highlights: [
      {
        title: '以研究节奏替代发布会语气',
        detail: '优先呈现运行假设、证据路径，以及评审者真正会检查的架构接缝。',
      },
      {
        title: '双主题下依然清晰稳定',
        detail: '设计 token 在白天亮屏与夜间深色模式中都保持可读性，不依赖噱头效果。',
      },
      {
        title: '为后续章节提供可复用图框',
        detail: '共享组件统一 hero、蓝图、atlas 与 research 区块，后续页面可直接接入。',
      },
    ],
    facts: [
      { label: '版本', value: '文档界面重建 / 任务 2' },
      { label: '读者', value: '高级工程师与面试评审者' },
      { label: '语气', value: '技术化、克制、强调证据' },
      { label: '范围', value: '算子、校验、基准、研究参考' },
    ],
    rail: [
      {
        title: '阅读模式',
        detail: '强左对齐与高密度图注让读者能在代码对照时快速扫描内容。',
      },
      {
        title: '图示纪律',
        detail: '图框、注释与版式节奏更像工程白皮书，而不是模板化营销页。',
      },
    ],
  },
}

const localeKey = computed<'en' | 'zh'>(() => (lang.value?.startsWith('zh') ? 'zh' : 'en'))

const content = computed<HeroContent>(() => {
  const fallback = localeContent[localeKey.value]
  return {
    eyebrow: props.eyebrow ?? fallback.eyebrow,
    title: props.title ?? fallback.title,
    dek: props.dek ?? fallback.dek,
    actions: (props.actions ?? fallback.actions).map((action) => ({
      ...action,
      href: withBase(action.href),
    })),
    highlights: props.highlights ?? fallback.highlights,
    facts: props.facts ?? fallback.facts,
    rail: props.rail ?? fallback.rail,
  }
})
</script>
