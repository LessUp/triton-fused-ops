<template>
  <figure class="docs-shell-stub">
    <div>
      <slot />
    </div>
    <figcaption v-if="title || caption">
      <strong v-if="title">{{ title }}</strong>
      <span v-if="caption">{{ caption }}</span>
    </figcaption>
  </figure>
</template>

<script setup lang="ts">
defineProps<{
  title?: string
  caption?: string
}>()
</script>
