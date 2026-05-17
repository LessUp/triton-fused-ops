<template>
  <section class="research-landscape" :aria-label="content.title">
    <div class="editorial-shell">
      <span class="editorial-kicker">{{ content.eyebrow }}</span>
      <h2 class="editorial-title">{{ content.title }}</h2>
      <p class="editorial-intro">{{ content.intro }}</p>
    </div>

    <div class="research-landscape__layout">
      <div class="research-landscape__clusters">
        <article
          v-for="cluster in content.clusters"
          :key="cluster.title"
          class="research-landscape__cluster"
        >
          <span class="research-landscape__period">{{ cluster.period }}</span>
          <div>
            <h3 class="research-landscape__cluster-title">{{ cluster.title }}</h3>
            <p>{{ cluster.summary }}</p>
            <div class="research-landscape__signals">
              <span
                v-for="signal in cluster.signals"
                :key="signal"
                class="research-landscape__signal"
              >
                {{ signal }}
              </span>
            </div>
          </div>
        </article>
      </div>

      <aside class="research-landscape__sidebar">
        <div
          v-for="panel in content.panels"
          :key="panel.title"
          class="research-landscape__sidebar-panel"
        >
          <strong>{{ panel.title }}</strong>
          <p class="editorial-caption">{{ panel.detail }}</p>
        </div>
      </aside>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'

interface ResearchCluster {
  period: string
  title: string
  summary: string
  signals: string[]
}

interface ResearchPanel {
  title: string
  detail: string
}

interface ResearchLandscapeContent {
  eyebrow: string
  title: string
  intro: string
  clusters: ResearchCluster[]
  panels: ResearchPanel[]
}

const props = defineProps<{
  eyebrow?: string
  title?: string
  intro?: string
  clusters?: ResearchCluster[]
  panels?: ResearchPanel[]
}>()

const { lang } = useData()

const localeContent: Record<'en' | 'zh', ResearchLandscapeContent> = {
  en: {
    eyebrow: 'Research Landscape',
    title: 'Positioning the library inside the broader inference conversation',
    intro: 'Use this component to connect implementation pages to the technical questions that matter in operator fusion, quantization, and deployment realism.',
    clusters: [
      {
        period: 'Track A',
        title: 'Kernel fusion as memory policy',
        summary: 'The strongest argument is rarely “more operations in one kernel”; it is disciplined removal of intermediate traffic and launch overhead.',
        signals: ['HBM avoidance', 'launch consolidation', 'shape-aware fusion'],
      },
      {
        period: 'Track B',
        title: 'Quantization as systems trade-off',
        summary: 'FP8 stories only convince when formats, scaling rules, and overflow behavior are explicit in both API and documentation.',
        signals: ['E4M3 / E5M2', 'scale management', 'error surfaces'],
      },
      {
        period: 'Track C',
        title: 'Benchmarking as evidence curation',
        summary: 'Performance sections must reveal how measurements were taken, what the baseline is, and where correctness gates sit.',
        signals: ['baseline framing', 'derived metrics', 'verification hooks'],
      },
    ],
    panels: [
      {
        title: 'Why it belongs in docs',
        detail: 'A whitepaper surface should tell readers how the project relates to external ideas, not just enumerate internal modules.',
      },
      {
        title: 'How to customize',
        detail: 'Swap clusters or sidebar panels per page to focus on a benchmark story, an architecture question, or a literature survey.',
      },
      {
        title: 'Visual stance',
        detail: 'A timeline-like arrangement gives rhythm without turning the page into a conference poster.',
      },
    ],
  },
  zh: {
    eyebrow: '研究图景',
    title: '把项目放回更广泛的推理系统讨论中',
    intro: '这个组件用于把实现页面连接到融合算子、量化路径与部署现实中的关键技术问题，而不是只罗列内部模块。',
    clusters: [
      {
        period: '线索 A',
        title: '算子融合本质上是内存策略',
        summary: '最有说服力的论点通常不是“一个 kernel 做更多事”，而是有纪律地消除中间结果与 launch 开销。',
        signals: ['避免 HBM 往返', '合并 launch', '按 shape 设计融合'],
      },
      {
        period: '线索 B',
        title: '量化是一种系统性权衡',
        summary: '只有当格式、缩放规则与溢出行为在 API 和文档中都被明确呈现，FP8 的叙事才可信。',
        signals: ['E4M3 / E5M2', '缩放管理', '误差界面'],
      },
      {
        period: '线索 C',
        title: 'Benchmark 是证据编排',
        summary: '性能章节必须交代测量方法、对照基线以及正确性闸门所在，否则数字缺乏说服力。',
        signals: ['基线语境', '派生指标', '验证钩子'],
      },
    ],
    panels: [
      {
        title: '为什么它属于文档',
        detail: '高质量白皮书界面不应只展示内部结构，还要说明项目与外部技术话题之间的关系。',
      },
      {
        title: '如何定制',
        detail: '不同章节可以替换 cluster 与侧栏内容，用于聚焦 benchmark、架构问题或研究综述。',
      },
      {
        title: '视觉姿态',
        detail: '带有时间线节奏的版式既能建立层次，也不会让页面变成会议海报。',
      },
    ],
  },
}

const localeKey = computed<'en' | 'zh'>(() => (lang.value?.startsWith('zh') ? 'zh' : 'en'))

const content = computed<ResearchLandscapeContent>(() => {
  const fallback = localeContent[localeKey.value]
  return {
    eyebrow: props.eyebrow ?? fallback.eyebrow,
    title: props.title ?? fallback.title,
    intro: props.intro ?? fallback.intro,
    clusters: props.clusters ?? fallback.clusters,
    panels: props.panels ?? fallback.panels,
  }
})
</script>
