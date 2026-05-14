<template>
  <div class="architecture-preview">
    <div class="layer-stack">
      <div
        v-for="layer in layers"
        :key="layer.name"
        class="layer"
        :class="layer.type"
        @click="handleClick(layer.link)"
      >
        <span class="layer-name">{{ layer.name }}</span>
        <span class="layer-modules mono">{{ layer.modules }}</span>
      </div>
    </div>
    <p class="layer-hint">{{ t.hint }}</p>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useData } from 'vitepress'

const { lang } = useData()

const i18n = {
  en: {
    hint: 'Click a layer to explore its documentation.',
  },
  zh: {
    hint: '点击层级查看对应文档。',
  },
}

const layerData = {
  en: [
    { name: 'API', modules: 'triton_ops.__init__', type: 'api', link: '/en/api/' },
    { name: 'Validation', modules: 'validation.py', type: 'validation', link: '/en/api/validation' },
    { name: 'Kernels', modules: 'kernels/*', type: 'gpu', link: '/en/api/kernels' },
    { name: 'Reference', modules: 'reference/*', type: 'cpu', link: '/en/internals/architecture' },
    { name: 'Tooling', modules: 'autotuner, benchmark', type: 'tooling', link: '/en/api/autotuner' },
  ],
  zh: [
    { name: 'API', modules: 'triton_ops.__init__', type: 'api', link: '/zh/api/' },
    { name: '校验层', modules: 'validation.py', type: 'validation', link: '/zh/api/validation' },
    { name: '算子层', modules: 'kernels/*', type: 'gpu', link: '/zh/api/kernels' },
    { name: '参考实现', modules: 'reference/*', type: 'cpu', link: '/zh/internals/architecture' },
    { name: '工具层', modules: 'autotuner, benchmark', type: 'tooling', link: '/zh/api/autotuner' },
  ],
}

const currentLang = computed(() => (lang.value?.startsWith('zh') ? 'zh' : 'en'))
const layers = computed(() => layerData[currentLang.value])
const t = computed(() => i18n[currentLang.value])

function handleClick(link) {
  if (link) {
    window.location.href = link
  }
}
</script>

<style scoped>
.architecture-preview {
  margin: 32px 0;
}

.layer-hint {
  text-align: center;
  font-size: 13px;
  color: var(--vp-c-text-3);
  margin-top: 16px;
}
</style>