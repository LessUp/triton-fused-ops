---
layout: home
hero:
  name: Triton Fused Ops
  text: ' '
  actions:
    - theme: brand
      text: English
      link: /en/
    - theme: alt
      text: 简体中文
      link: /zh/
---

<script setup>
import { onMounted } from 'vue'
import { useRouter, withBase } from 'vitepress'

onMounted(() => {
  const router = useRouter()
  const lang = navigator.language || 'en-US'
  const target = lang.toLowerCase().startsWith('zh') ? withBase('/zh/') : withBase('/en/')
  if (lang.toLowerCase().startsWith('zh')) {
    router.go(target)
  } else {
    router.go(target)
  }
})
</script>
