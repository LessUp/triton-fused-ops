<template>
  <div class="kernel-showcase">
    <div class="kernel-grid">
      <a v-for="kernel in kernels" :key="kernel.id" :href="kernel.link" class="kernel-card">
        <span class="tag">{{ kernel.tag }}</span>
        <h3>{{ kernel.name }}</h3>
        <p>{{ kernel.description }}</p>
        <div class="metrics">
          <div class="metric">
            <span class="icon">⚡</span>
            <span class="value">{{ kernel.speedup }}</span>
          </div>
          <div class="metric">
            <span class="icon">💾</span>
            <span class="value">{{ kernel.memory }}</span>
          </div>
        </div>
      </a>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useData } from 'vitepress'

const { lang } = useData()

const kernelData = {
  en: [
    {
      id: 'rmsnorm-rope',
      tag: 'FUSION',
      name: 'RMSNorm + RoPE',
      description: 'Fused RMSNorm and Rotary Position Embedding in a single kernel launch. Eliminates intermediate HBM round-trips.',
      speedup: '~3×',
      memory: '-40%',
      link: '/en/api/kernels/#fused-rmsnorm-rope',
    },
    {
      id: 'gated-mlp',
      tag: 'FUSION',
      name: 'Gated MLP',
      description: 'Fused gate projection, activation, and up projection. Supports SiLU and GELU activations.',
      speedup: '~1.5×',
      memory: '-25%',
      link: '/en/api/kernels/#fused-gated-mlp',
    },
    {
      id: 'fp8-gemm',
      tag: 'QUANTIZATION',
      name: 'FP8 GEMM',
      description: 'E4M3/E5M2-compatible FP8 quantized GEMM with explicit scale management and overflow handling.',
      speedup: '~1.3×',
      memory: '-50%',
      link: '/en/api/kernels/#fp8-gemm',
    },
    {
      id: 'fp8-quantize',
      tag: 'QUANTIZATION',
      name: 'FP8 Quantize',
      description: 'FP8 quantization with per-tensor scaling. Vectorized implementation for high throughput.',
      speedup: 'N/A',
      memory: '-75%',
      link: '/en/api/quantization/',
    },
  ],
  zh: [
    {
      id: 'rmsnorm-rope',
      tag: '融合算子',
      name: 'RMSNorm + RoPE',
      description: '单次 kernel launch 完成 RMSNorm 与旋转位置编码，消除中间结果的 HBM 往返。',
      speedup: '~3×',
      memory: '-40%',
      link: '/zh/api/kernels/#fused-rmsnorm-rope',
    },
    {
      id: 'gated-mlp',
      tag: '融合算子',
      name: 'Gated MLP',
      description: '融合门控投影、激活函数与上投影。支持 SiLU 和 GELU 激活函数。',
      speedup: '~1.5×',
      memory: '-25%',
      link: '/zh/api/kernels/#fused-gated-mlp',
    },
    {
      id: 'fp8-gemm',
      tag: '量化算子',
      name: 'FP8 GEMM',
      description: 'E4M3/E5M2 兼容的 FP8 量化 GEMM，显式缩放因子管理与溢出处理。',
      speedup: '~1.3×',
      memory: '-50%',
      link: '/zh/api/kernels/#fp8-gemm',
    },
    {
      id: 'fp8-quantize',
      tag: '量化算子',
      name: 'FP8 量化',
      description: 'FP8 量化算子，支持 per-tensor 缩放。向量化实现，高吞吐量。',
      speedup: 'N/A',
      memory: '-75%',
      link: '/zh/api/quantization/',
    },
  ],
}

const currentLang = computed(() => lang.value?.startsWith('zh') ? 'zh' : 'en')
const kernels = computed(() => kernelData[currentLang.value])
</script>
