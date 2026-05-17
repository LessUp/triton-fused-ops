<template>
  <section class="kernel-atlas" :aria-label="content.title">
    <div class="editorial-shell">
      <span class="editorial-kicker">{{ content.eyebrow }}</span>
      <h2 class="editorial-title">{{ content.title }}</h2>
      <p class="editorial-intro">{{ content.intro }}</p>
    </div>

    <div class="kernel-atlas__layout">
      <article class="kernel-atlas__featured">
        <span class="kernel-atlas__eyebrow">{{ content.featureLabel }}</span>
        <div>
          <h3 class="kernel-atlas__featured-title">{{ featured.name }}</h3>
          <p>{{ featured.summary }}</p>
        </div>
        <div class="kernel-atlas__facts">
          <div v-for="fact in featured.facts" :key="fact.label" class="kernel-atlas__fact">
            <strong>{{ fact.label }}</strong>
            <span class="editorial-caption">{{ fact.value }}</span>
          </div>
        </div>
        <a :href="featured.href" class="reader-tracks__link">
          <span>{{ content.ctaLabel }}</span>
          <span class="editorial-link-arrow" aria-hidden="true">→</span>
        </a>
      </article>

      <div class="kernel-atlas__list">
        <article v-for="(family, index) in secondaryFamilies" :key="family.name" class="kernel-atlas__item">
          <div class="kernel-atlas__item-head">
            <span class="kernel-atlas__item-index">{{ String(index + 2).padStart(2, '0') }}</span>
            <h3 class="kernel-atlas__item-title">{{ family.name }}</h3>
          </div>
          <p class="editorial-caption">{{ family.summary }}</p>
          <div class="kernel-atlas__item-meta">
            <span v-for="tag in family.tags" :key="tag" class="kernel-atlas__chip">{{ tag }}</span>
          </div>
        </article>
      </div>
    </div>

    <div class="kernel-atlas__notes">
      <div v-for="note in content.notes" :key="note.title" class="kernel-atlas__note">
        <span class="kernel-atlas__footnote-label">{{ note.title }}</span>
        <strong>{{ note.heading }}</strong>
        <p class="editorial-caption">{{ note.detail }}</p>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'

interface AtlasFact {
  label: string
  value: string
}

interface KernelFamily {
  name: string
  summary: string
  href: string
  facts: AtlasFact[]
  tags: string[]
}

interface AtlasNote {
  title: string
  heading: string
  detail: string
}

interface KernelAtlasContent {
  eyebrow: string
  title: string
  intro: string
  featureLabel: string
  ctaLabel: string
  families: KernelFamily[]
  notes: AtlasNote[]
}

const props = defineProps<{
  eyebrow?: string
  title?: string
  intro?: string
  featureLabel?: string
  ctaLabel?: string
  families?: KernelFamily[]
  notes?: AtlasNote[]
}>()

const { lang } = useData()

const localeContent: Record<'en' | 'zh', KernelAtlasContent> = {
  en: {
    eyebrow: 'Kernel Atlas',
    title: 'A reusable map of the library’s kernel families',
    intro: 'Use the atlas when a page needs to compare user-facing families without collapsing them into repetitive marketing cards.',
    featureLabel: 'Featured family',
    ctaLabel: 'Inspect family docs',
    families: [
      {
        name: 'Fused RMSNorm + RoPE',
        summary: 'A launch path optimized around normalization and rotary embedding in one movement through memory.',
        href: '/en/kernel-families/',
        facts: [
          { label: 'Primary use', value: 'Attention pre-processing' },
          { label: 'Review angle', value: 'HBM traffic removal and shape discipline' },
          { label: 'Validation seam', value: 'Reference compute + launcher checks' },
          { label: 'Narrative role', value: 'The clearest fusion story in the library' },
        ],
        tags: ['fusion', 'rope', 'rmsnorm'],
      },
      {
        name: 'Fused Gated MLP',
        summary: 'Combines gate projection, activation, and up projection to reduce traffic in dense transformer blocks.',
        href: '/en/kernel-families/',
        facts: [],
        tags: ['fusion', 'swiglu', 'geglu'],
      },
      {
        name: 'FP8 GEMM',
        summary: 'Quantized matrix multiplication with explicit scale management and operational care around precision modes.',
        href: '/en/kernel-families/',
        facts: [],
        tags: ['fp8', 'gemm', 'quantization'],
      },
      {
        name: 'FP8 Quantization Utilities',
        summary: 'The conversion and scaling utilities that make the FP8 pathway auditable instead of magical.',
        href: '/en/kernel-families/',
        facts: [],
        tags: ['utility', 'scales', 'formats'],
      },
    ],
    notes: [
      {
        title: 'Operator posture',
        heading: 'Each family should advertise one dominant claim',
        detail: 'This component keeps the page honest by forcing a single operating story per family rather than mixing every feature together.',
      },
      {
        title: 'Comparison rhythm',
        heading: 'One featured block, several satellite entries',
        detail: 'The layout avoids the fatigue of six identical cards while preserving scan-ability for side-by-side reading.',
      },
      {
        title: 'Extension point',
        heading: 'Custom families can be passed in as props',
        detail: 'Later pages can swap in section-specific data without changing the framing system.',
      },
    ],
  },
  zh: {
    eyebrow: 'Kernel Atlas',
    title: '可复用的算子族地图',
    intro: '当页面需要对比用户可见的算子族时，用 atlas 而不是模板化营销卡片，可以更清楚地表达差异。',
    featureLabel: '重点算子族',
    ctaLabel: '查看算子文档',
    families: [
      {
        name: 'Fused RMSNorm + RoPE',
        summary: '围绕一次内存通路完成归一化与旋转位置编码，是库中最具代表性的融合路径。',
        href: '/zh/kernel-families/',
        facts: [
          { label: '主要用途', value: '注意力前处理' },
          { label: '评审重点', value: 'HBM 往返削减与 shape 约束' },
          { label: '验证接缝', value: '参考实现与 launcher 校验' },
          { label: '叙事角色', value: '最清晰的融合价值示例' },
        ],
        tags: ['fusion', 'rope', 'rmsnorm'],
      },
      {
        name: 'Fused Gated MLP',
        summary: '把 gate projection、激活与 up projection 合并，减少 Transformer dense block 中的数据往返。',
        href: '/zh/kernel-families/',
        facts: [],
        tags: ['fusion', 'swiglu', 'geglu'],
      },
      {
        name: 'FP8 GEMM',
        summary: '显式管理缩放因子和精度模式的量化矩阵乘法，强调工程可审查性。',
        href: '/zh/kernel-families/',
        facts: [],
        tags: ['fp8', 'gemm', 'quantization'],
      },
      {
        name: 'FP8 Quantization Utilities',
        summary: '为 FP8 路径提供可解释的转换与缩放工具，避免“黑盒量化”的观感。',
        href: '/zh/kernel-families/',
        facts: [],
        tags: ['utility', 'scales', 'formats'],
      },
    ],
    notes: [
      {
        title: '算子姿态',
        heading: '每个算子族只强调一个主叙事',
        detail: '组件通过结构约束，让页面为每个算子族给出一个主张，而不是把所有特性堆在一起。',
      },
      {
        title: '比较节奏',
        heading: '一个重点区块，多个卫星条目',
        detail: '避免六张完全相同的卡片造成疲劳，同时保留快速扫描能力。',
      },
      {
        title: '扩展点',
        heading: '支持通过 props 注入定制数据',
        detail: '后续章节可以替换内容，但继续复用 atlas 的版式框架。',
      },
    ],
  },
}

const localeKey = computed<'en' | 'zh'>(() => (lang.value?.startsWith('zh') ? 'zh' : 'en'))

const content = computed<KernelAtlasContent>(() => {
  const fallback = localeContent[localeKey.value]
  const families = props.families && props.families.length > 0 ? props.families : fallback.families
  return {
    eyebrow: props.eyebrow ?? fallback.eyebrow,
    title: props.title ?? fallback.title,
    intro: props.intro ?? fallback.intro,
    featureLabel: props.featureLabel ?? fallback.featureLabel,
    ctaLabel: props.ctaLabel ?? fallback.ctaLabel,
    families: families.map((family) => ({
      ...family,
      href: withBase(family.href),
    })),
    notes: props.notes ?? fallback.notes,
  }
})

const featured = computed(() => content.value.families[0])
const secondaryFamilies = computed(() => content.value.families.slice(1))
</script>
