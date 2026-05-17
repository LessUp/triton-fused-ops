<template>
  <section class="reader-tracks" :aria-label="content.title">
    <div class="editorial-shell">
      <span class="editorial-kicker">{{ content.eyebrow }}</span>
      <h2 class="editorial-title">{{ content.title }}</h2>
      <p class="editorial-intro">{{ content.intro }}</p>
    </div>

    <div class="reader-tracks__grid">
      <article class="reader-tracks__primary">
        <div class="reader-tracks__primary-topline">
          <span class="reader-tracks__index">01</span>
          <span class="reader-tracks__effort">{{ primaryTrack.duration }}</span>
        </div>
        <div>
          <h3 class="reader-tracks__name">{{ primaryTrack.name }}</h3>
          <p class="reader-tracks__summary">{{ primaryTrack.summary }}</p>
        </div>
        <div class="reader-tracks__checkpoints">
          <div
            v-for="checkpoint in primaryTrack.checkpoints"
            :key="checkpoint.title"
            class="reader-tracks__checkpoint"
          >
            <strong>{{ checkpoint.title }}</strong>
            <span class="editorial-caption">{{ checkpoint.detail }}</span>
          </div>
        </div>
        <div class="reader-tracks__links">
          <a :href="primaryTrack.href" class="reader-tracks__link">
            <span>{{ content.ctaLabel }}</span>
            <span class="editorial-link-arrow" aria-hidden="true">→</span>
          </a>
        </div>
      </article>

      <div class="reader-tracks__stack">
        <article
          v-for="(track, index) in secondaryTracks"
          :key="track.name"
          class="reader-tracks__stack-item"
        >
          <div class="reader-tracks__stack-topline">
            <span class="reader-tracks__index">{{ String(index + 2).padStart(2, '0') }}</span>
            <span class="reader-tracks__effort">{{ track.duration }}</span>
          </div>
          <h3 class="reader-tracks__name">{{ track.name }}</h3>
          <p class="reader-tracks__summary">{{ track.summary }}</p>
          <ul class="editorial-meta-list">
            <li v-for="point in track.points" :key="point">{{ point }}</li>
          </ul>
          <a :href="track.href" class="reader-tracks__link">
            <span>{{ content.ctaLabel }}</span>
            <span class="editorial-link-arrow" aria-hidden="true">→</span>
          </a>
        </article>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'

interface TrackCheckpoint {
  title: string
  detail: string
}

interface ReaderTrack {
  name: string
  summary: string
  duration: string
  href: string
  checkpoints?: TrackCheckpoint[]
  points?: string[]
}

interface ReaderTracksContent {
  eyebrow: string
  title: string
  intro: string
  ctaLabel: string
  tracks: ReaderTrack[]
}

const props = defineProps<{
  eyebrow?: string
  title?: string
  intro?: string
  ctaLabel?: string
  tracks?: ReaderTrack[]
}>()

const { lang } = useData()

const localeContent: Record<'en' | 'zh', ReaderTracksContent> = {
  en: {
    eyebrow: 'Reading Tracks',
    title: 'Three entry paths for three reviewer mindsets',
    intro: 'Later pages can drop this component in when a section needs to route readers quickly without flattening everything into identical cards.',
    ctaLabel: 'Open track',
    tracks: [
      {
        name: 'Evaluator track',
        summary: 'For reviewers who need fast confidence that the architecture is disciplined, testable, and honest about boundaries.',
        duration: '8 min read',
        href: '/en/architecture-lab/',
        checkpoints: [
          {
            title: 'Start with system boundaries',
            detail: 'Scan how public APIs, validation, compute references, and Triton kernels are separated.',
          },
          {
            title: 'Inspect proof surfaces',
            detail: 'Follow where correctness is asserted: compute references, benchmark checks, and tuning constraints.',
          },
        ],
      },
      {
        name: 'Maintainer track',
        summary: 'For engineers touching implementation details, module seams, or extension points.',
        duration: '12 min read',
        href: '/en/guides/',
        points: [
          'Map validation to launchers before changing kernels.',
          'Understand where latency concerns stop and derived metrics begin.',
          'Treat shared figures as stable documentation primitives.',
        ],
      },
      {
        name: 'Research track',
        summary: 'For readers comparing the project to broader inference, quantization, and operator-fusion ideas.',
        duration: '10 min read',
        href: '/en/reference-research/',
        points: [
          'Trace kernel families against deployment constraints.',
          'Read performance claims with the benchmark context attached.',
          'Position Triton choices against the external landscape.',
        ],
      },
    ],
  },
  zh: {
    eyebrow: '阅读路径',
    title: '为不同评审心智准备的三条入口',
    intro: '后续页面可直接复用这个组件，在不落入单一卡片栅格模板的前提下，快速引导读者进入正确的章节。',
    ctaLabel: '进入路径',
    tracks: [
      {
        name: '评审者路径',
        summary: '适合需要快速判断架构是否克制、可验证、是否诚实呈现边界的读者。',
        duration: '约 8 分钟',
        href: '/zh/architecture-lab/',
        checkpoints: [
          {
            title: '先看系统边界',
            detail: '快速理解公开 API、输入校验、CPU 参考实现与 Triton kernel 如何被分层。',
          },
          {
            title: '再看证据界面',
            detail: '沿着 compute reference、benchmark 校验与调优约束，确认正确性如何被证明。',
          },
        ],
      },
      {
        name: '维护者路径',
        summary: '适合准备修改实现细节、模块接缝或扩展能力的工程师。',
        duration: '约 12 分钟',
        href: '/zh/guides/',
        points: [
          '动 kernel 之前先定位 validation 与 launcher 的契约。',
          '明确 latency 关注点与派生性能指标的边界。',
          '把共享图框当作文档基础设施而不是一次性装饰。',
        ],
      },
      {
        name: '研究路径',
        summary: '适合把项目放进更广泛推理、量化与融合算子背景中理解的读者。',
        duration: '约 10 分钟',
        href: '/zh/reference-research/',
        points: [
          '将算子族与实际部署约束对应起来。',
          '在 benchmark 语境下解读性能陈述。',
          '把 Triton 设计选择放回外部研究图景里观察。',
        ],
      },
    ],
  },
}

const localeKey = computed<'en' | 'zh'>(() => (lang.value?.startsWith('zh') ? 'zh' : 'en'))

const content = computed<ReaderTracksContent>(() => {
  const fallback = localeContent[localeKey.value]
  const tracks = props.tracks && props.tracks.length > 0 ? props.tracks : fallback.tracks
  return {
    eyebrow: props.eyebrow ?? fallback.eyebrow,
    title: props.title ?? fallback.title,
    intro: props.intro ?? fallback.intro,
    ctaLabel: props.ctaLabel ?? fallback.ctaLabel,
    tracks: tracks.map((track) => ({
      ...track,
      href: withBase(track.href),
    })),
  }
})

const primaryTrack = computed(() => content.value.tracks[0])
const secondaryTracks = computed(() => content.value.tracks.slice(1))
</script>
