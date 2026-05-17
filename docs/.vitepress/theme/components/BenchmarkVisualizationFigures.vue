<template>
  <section class="benchmark-visualization">
    <div class="benchmark-visualization__section">
      <h2>{{ content.latency.heading }}</h2>
      <FigureFrame
        class="benchmark-figure"
        tone="accent"
        :label="content.latency.label"
        :title="content.latency.title"
        :caption="content.latency.caption"
      >
        <PerformanceChart :data="latencyData" :height="320" value-unit="µs" y-axis-label="Latency (µs)" />
      </FigureFrame>
    </div>

    <div class="benchmark-visualization__section">
      <h2>{{ content.speedup.heading }}</h2>
      <FigureFrame
        class="benchmark-figure"
        :label="content.speedup.label"
        :title="content.speedup.title"
        :caption="content.speedup.caption"
      >
        <PerformanceChart :data="speedupData" :height="320" show-speedup />
      </FigureFrame>
    </div>

    <div class="benchmark-visualization__section">
      <h2>{{ content.memory.heading }}</h2>
      <FigureFrame
        class="benchmark-figure"
        :label="content.memory.label"
        :title="content.memory.title"
        :caption="content.memory.caption"
      >
        <div class="viz-panel">
          <div class="memory-traffic" :aria-label="content.memory.ariaLabel" role="img">
            <div
              v-for="stack in content.memory.stacks"
              :key="stack.label"
              class="memory-traffic__stack"
            >
              <div class="memory-traffic__column">
                <div
                  v-for="segment in stack.segments"
                  :key="`${stack.label}-${segment.key}`"
                  class="memory-traffic__segment"
                  :class="`memory-traffic__segment--${segment.key}`"
                  :style="{ height: `${segment.height}px` }"
                >
                  <span>{{ segment.label }}</span>
                </div>
              </div>
              <span class="memory-traffic__label">{{ stack.label }}</span>
            </div>
          </div>

          <div class="viz-legend" aria-hidden="true">
            <span
              v-for="item in content.memory.legend"
              :key="item.label"
              class="viz-legend__item"
            >
              <span class="viz-legend__swatch" :class="`viz-legend__swatch--${item.key}`"></span>
              {{ item.label }}
            </span>
          </div>
        </div>
      </FigureFrame>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'
import FigureFrame from './FigureFrame.vue'
import PerformanceChart from './PerformanceChart.vue'

interface Segment {
  key: 'read' | 'write' | 'register'
  label: string
  height: number
}

interface Stack {
  label: string
  segments: Segment[]
}

interface MemoryContent {
  heading: string
  label: string
  title: string
  caption: string
  ariaLabel: string
  legend: Array<{ key: Segment['key']; label: string }>
  stacks: Stack[]
}

interface VisualizationCopy {
  latency: {
    heading: string
    label: string
    title: string
    caption: string
  }
  speedup: {
    heading: string
    label: string
    title: string
    caption: string
  }
  memory: MemoryContent
}

const { lang } = useData()

const latencyData = [
  { label: '128', pytorch: 32, triton: 14 },
  { label: '512', pytorch: 44, triton: 20 },
  { label: '1024', pytorch: 68, triton: 28 },
  { label: '2048', pytorch: 80, triton: 38 },
  { label: '4096', pytorch: 80, triton: 50 },
  { label: '8192', pytorch: 80, triton: 60 },
]

const speedupData = [
  { label: '1', pytorch: 1.2, triton: 1 },
  { label: '4', pytorch: 1.55, triton: 1 },
  { label: '16', pytorch: 1.95, triton: 1 },
  { label: '64', pytorch: 2.45, triton: 1 },
  { label: '128', pytorch: 2.7, triton: 1 },
]

const localeContent: Record<'en' | 'zh', VisualizationCopy> = {
  en: {
    latency: {
      heading: 'Latency vs Sequence Length (`fused_rmsnorm_rope`)',
      label: 'Figure 01',
      title: 'Latency trend stays legible across theme switches',
      caption:
        'Sequence length versus latency (µs) at batch=2, hidden_dim=4096. Triton keeps the fused path consistently below the unfused PyTorch baseline.',
    },
    speedup: {
      heading: 'Speedup vs Batch Size',
      label: 'Figure 02',
      title: 'Speedup plot uses tokenized axes, fills, and baseline cues',
      caption: 'Batch size versus speedup ratio for fused_rmsnorm_rope at seq_len=2048.',
    },
    memory: {
      heading: 'Memory Traffic Breakdown',
      label: 'Figure 03',
      title: 'Memory traffic diagram now follows the same tokenized figure language',
      caption: 'Per-forward-pass traffic comparison for fused_rmsnorm_rope at batch=2, seq_len=2048.',
      ariaLabel: 'Stacked memory traffic comparison between fused and unfused execution',
      legend: [
        { key: 'read', label: 'HBM Read' },
        { key: 'write', label: 'HBM Write' },
        { key: 'register', label: 'Register Traffic' },
      ],
      stacks: [
        {
          label: 'Fused',
          segments: [
            { key: 'read', label: 'Read', height: 60 },
            { key: 'write', label: 'Write', height: 20 },
            { key: 'register', label: 'Reg', height: 80 },
          ],
        },
        {
          label: 'Unfused',
          segments: [
            { key: 'read', label: 'Read', height: 100 },
            { key: 'write', label: 'Write', height: 60 },
            { key: 'register', label: 'Reg', height: 10 },
          ],
        },
      ],
    },
  },
  zh: {
    latency: {
      heading: 'Latency vs Sequence Length (`fused_rmsnorm_rope`)',
      label: '图 01',
      title: '延迟图在明暗主题之间都保持清晰层次',
      caption:
        'Sequence length 与延迟（µs）的关系，batch=2、hidden_dim=4096。Triton 的融合路径持续低于未融合的 PyTorch 基线。',
    },
    speedup: {
      heading: 'Speedup vs Batch Size',
      label: '图 02',
      title: '加速比曲线改为使用主题 token 的坐标轴、填充和基线提示',
      caption: 'fused_rmsnorm_rope 在 seq_len=2048 条件下的 batch size 与加速比关系。',
    },
    memory: {
      heading: '内存流量拆解',
      label: '图 03',
      title: '内存流量图也统一到同一套主题安全视觉语言',
      caption: '单次前向的流量对比，条件为 fused_rmsnorm_rope、batch=2、seq_len=2048。',
      ariaLabel: '融合与未融合执行的堆叠内存流量对比',
      legend: [
        { key: 'read', label: 'HBM Read' },
        { key: 'write', label: 'HBM Write' },
        { key: 'register', label: 'Register Traffic' },
      ],
      stacks: [
        {
          label: '融合',
          segments: [
            { key: 'read', label: 'Read', height: 60 },
            { key: 'write', label: 'Write', height: 20 },
            { key: 'register', label: 'Reg', height: 80 },
          ],
        },
        {
          label: '未融合',
          segments: [
            { key: 'read', label: 'Read', height: 100 },
            { key: 'write', label: 'Write', height: 60 },
            { key: 'register', label: 'Reg', height: 10 },
          ],
        },
      ],
    },
  },
}

const content = computed<VisualizationCopy>(() =>
  lang.value?.startsWith('zh') ? localeContent.zh : localeContent.en
)
</script>

<style scoped>
.benchmark-visualization {
  display: grid;
  gap: 2rem;
}

.benchmark-visualization__section {
  display: grid;
  gap: 1rem;
}

.benchmark-figure {
  --viz-accent: var(--editorial-signal);
  --viz-accent-soft: color-mix(in srgb, var(--editorial-signal) 18%, transparent);
  --viz-baseline: color-mix(in srgb, var(--editorial-accent) 68%, var(--editorial-panel));
  --viz-axis: var(--editorial-rule-strong);
  --viz-grid: var(--editorial-rule);
  --viz-text: var(--vp-c-text-2);
  --viz-read: var(--editorial-accent);
  --viz-write: color-mix(in srgb, var(--editorial-accent) 48%, var(--editorial-signal));
  --viz-register: var(--editorial-signal);
  --viz-segment-text: var(--vp-c-bg);
}

.viz-panel {
  display: grid;
  gap: 1rem;
}

.viz-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.85rem 1.2rem;
  color: var(--viz-text);
  font-size: 0.9rem;
}

.viz-legend__item {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
}

.viz-legend__swatch {
  width: 0.9rem;
  height: 0.9rem;
  border-radius: 999px;
  background: var(--viz-axis);
}

.viz-legend__swatch--read {
  background: var(--viz-read);
}

.viz-legend__swatch--write {
  background: var(--viz-write);
}

.viz-legend__swatch--register {
  background: var(--viz-register);
}

.memory-traffic {
  display: flex;
  justify-content: center;
  gap: clamp(1.5rem, 8vw, 4rem);
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--viz-grid);
}

.memory-traffic__stack {
  display: grid;
  justify-items: center;
  gap: 0.75rem;
}

.memory-traffic__column {
  width: 88px;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
}

.memory-traffic__segment {
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--viz-segment-text);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.memory-traffic__segment--read {
  background: linear-gradient(180deg, color-mix(in srgb, var(--viz-read) 74%, white), var(--viz-read));
  border-radius: 0.8rem 0.8rem 0 0;
}

.memory-traffic__segment--write {
  background: linear-gradient(180deg, color-mix(in srgb, var(--viz-write) 72%, white), var(--viz-write));
}

.memory-traffic__segment--register {
  background: linear-gradient(
    180deg,
    color-mix(in srgb, var(--viz-register) 76%, white),
    var(--viz-register)
  );
  border-radius: 0 0 0.8rem 0.8rem;
}

.memory-traffic__label {
  color: var(--vp-c-text-1);
  font-size: 0.92rem;
  font-weight: 700;
}

:global(.dark) .benchmark-figure {
  --viz-segment-text: var(--editorial-paper);
}

@media (max-width: 640px) {
  .memory-traffic {
    gap: 1.25rem;
  }

  .memory-traffic__column {
    width: 74px;
  }

  .viz-legend {
    font-size: 0.82rem;
  }
}
</style>
