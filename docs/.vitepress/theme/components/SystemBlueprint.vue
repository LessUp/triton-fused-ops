<template>
  <section class="system-blueprint" :aria-label="content.title">
    <div class="editorial-shell">
      <span class="editorial-kicker">{{ content.eyebrow }}</span>
      <h2 class="editorial-title">{{ content.title }}</h2>
      <p class="editorial-intro">{{ content.intro }}</p>
    </div>

    <div class="system-blueprint__layout">
      <div class="system-blueprint__layers">
        <article
          v-for="(layer, index) in content.layers"
          :key="layer.name"
          class="system-blueprint__layer"
        >
          <span class="system-blueprint__step">{{ String(index + 1).padStart(2, '0') }}</span>
          <div>
            <h3 class="system-blueprint__layer-title">{{ layer.name }}</h3>
            <div class="system-blueprint__modules mono">{{ layer.modules }}</div>
            <p>{{ layer.summary }}</p>
            <div class="system-blueprint__checkpoints">
              <span
                v-for="checkpoint in layer.checkpoints"
                :key="checkpoint"
                class="system-blueprint__checkpoint"
              >
                {{ checkpoint }}
              </span>
            </div>
          </div>
        </article>
      </div>

      <aside class="system-blueprint__sidebar">
        <div
          v-for="panel in content.panels"
          :key="panel.title"
          class="system-blueprint__sidebar-panel"
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

interface BlueprintLayer {
  name: string
  modules: string
  summary: string
  checkpoints: string[]
}

interface BlueprintPanel {
  title: string
  detail: string
}

interface BlueprintContent {
  eyebrow: string
  title: string
  intro: string
  layers: BlueprintLayer[]
  panels: BlueprintPanel[]
}

const props = defineProps<{
  eyebrow?: string
  title?: string
  intro?: string
  layers?: BlueprintLayer[]
  panels?: BlueprintPanel[]
}>()

const { lang } = useData()

const localeContent: Record<'en' | 'zh', BlueprintContent> = {
  en: {
    eyebrow: 'System Blueprint',
    title: 'A serial view of how the library keeps responsibilities separated',
    intro: 'This blueprint is meant for section pages that need an instant architecture read without dumping readers into source trees or giant diagrams.',
    layers: [
      {
        name: 'Public API surface',
        modules: 'triton_ops/__init__.py',
        summary: 'Expose only stable, user-facing launchers and wrappers while keeping internal Triton kernels private.',
        checkpoints: ['stable imports', 'minimal surface', 'module wrappers'],
      },
      {
        name: 'Validation contracts',
        modules: 'triton_ops/validation.py',
        summary: 'Centralize dtype, shape, contiguity, and scalar checks so launchers stay predictable.',
        checkpoints: ['shape rules', 'dtype guards', 'device constraints'],
      },
      {
        name: 'Kernel and compute execution',
        modules: 'triton_ops/kernels/* + triton_ops/compute/*',
        summary: 'Keep fast-path GPU implementation and CPU-verifiable reference logic adjacent but not entangled.',
        checkpoints: ['kernel launchers', 'reference math', 'testability'],
      },
      {
        name: 'Measurement and tuning',
        modules: 'benchmark/* + autotuner/*',
        summary: 'Separate latency search from reporting language so claims remain measurable and reviewable.',
        checkpoints: ['latency tuning', 'correctness checks', 'report framing'],
      },
    ],
    panels: [
      {
        title: 'Why this view matters',
        detail: 'Interview-level reviewers want to know where correctness lives before they care about throughput claims.',
      },
      {
        title: 'Reusable by design',
        detail: 'Pages can override layers and side notes through props when focusing on one subsystem or a single kernel family.',
      },
      {
        title: 'Reading cue',
        detail: 'Treat the numbered stack as a causal chain: public promise → contract enforcement → execution path → measurement.',
      },
    ],
  },
  zh: {
    eyebrow: '系统蓝图',
    title: '串行展示库如何保持职责边界',
    intro: '这个蓝图适合放在章节页中，让读者迅速理解架构层次，而不必先钻进源码树或超大流程图。',
    layers: [
      {
        name: '公开 API 界面',
        modules: 'triton_ops/__init__.py',
        summary: '只暴露稳定的用户入口与包装器，内部 Triton kernel 保持私有。',
        checkpoints: ['稳定导入', '最小公开面', '模块包装器'],
      },
      {
        name: '输入校验契约',
        modules: 'triton_ops/validation.py',
        summary: '把 dtype、shape、contiguity 与标量检查集中起来，让 launcher 保持可预期。',
        checkpoints: ['shape 规则', 'dtype 守卫', 'device 约束'],
      },
      {
        name: 'Kernel 与参考执行',
        modules: 'triton_ops/kernels/* + triton_ops/compute/*',
        summary: '让 GPU 快路径与 CPU 可验证参考实现相邻但不纠缠，便于验证与维护。',
        checkpoints: ['kernel launcher', '参考数学', '可测试性'],
      },
      {
        name: '测量与调优',
        modules: 'benchmark/* + autotuner/*',
        summary: '把 latency 搜索与报告叙事拆开，确保性能陈述可以被测量与复核。',
        checkpoints: ['延迟调优', '正确性校验', '报告框架'],
      },
    ],
    panels: [
      {
        title: '为什么值得这样看',
        detail: '高级评审通常先确认正确性和边界在哪里，再决定是否相信吞吐与性能数字。',
      },
      {
        title: '为复用而生',
        detail: '后续页面可以通过 props 替换层级和侧栏说明，用于聚焦某个子系统或单个算子族。',
      },
      {
        title: '阅读提示',
        detail: '把编号顺序理解成因果链：公开承诺 → 契约执行 → 运行路径 → 测量与报告。',
      },
    ],
  },
}

const localeKey = computed<'en' | 'zh'>(() => (lang.value?.startsWith('zh') ? 'zh' : 'en'))

const content = computed<BlueprintContent>(() => {
  const fallback = localeContent[localeKey.value]
  return {
    eyebrow: props.eyebrow ?? fallback.eyebrow,
    title: props.title ?? fallback.title,
    intro: props.intro ?? fallback.intro,
    layers: props.layers ?? fallback.layers,
    panels: props.panels ?? fallback.panels,
  }
})
</script>
