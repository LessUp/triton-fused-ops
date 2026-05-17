<template>
  <figure class="figure-frame" :class="toneClass">
    <header v-if="hasHeader" class="figure-frame__header">
      <div>
        <span class="figure-frame__label">{{ resolvedLabel }}</span>
        <h3 v-if="title" class="figure-frame__title">{{ title }}</h3>
      </div>
      <slot name="meta" />
    </header>

    <div class="figure-frame__body">
      <slot />
    </div>

    <figcaption v-if="caption || credit" class="figure-frame__footer">
      <span v-if="caption" class="editorial-caption">{{ caption }}</span>
      <span v-if="credit" class="figure-frame__credit">{{ credit }}</span>
    </figcaption>
  </figure>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'

const props = withDefaults(
  defineProps<{
    label?: string
    title?: string
    caption?: string
    credit?: string
    index?: string | number
    tone?: 'default' | 'quiet' | 'accent'
  }>(),
  {
    tone: 'default',
  },
)

const { lang } = useData()

const localePrefix = computed(() => (lang.value?.startsWith('zh') ? '图' : 'Figure'))

const resolvedLabel = computed(() => {
  if (props.label) {
    return props.label
  }
  if (props.index === undefined) {
    return localePrefix.value
  }
  return `${localePrefix.value} ${props.index}`
})

const hasHeader = computed(() => Boolean(resolvedLabel.value || props.title))
const toneClass = computed(() => ({
  'figure-frame--quiet': props.tone === 'quiet',
  'figure-frame--accent': props.tone === 'accent',
}))
</script>
