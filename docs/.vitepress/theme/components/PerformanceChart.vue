<template>
  <div class="perf-chart-container">
    <canvas ref="chartRef"></canvas>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, computed } from 'vue'
import { useData } from 'vitepress'
import {
  Chart,
  BarController,
  CategoryScale,
  LinearScale,
  BarElement,
  Tooltip,
  Legend,
} from 'chart.js'

// Register Chart.js components
Chart.register(BarController, CategoryScale, LinearScale, BarElement, Tooltip, Legend)

const props = defineProps({
  // Array of { label: string, pytorch: number, triton: number }
  data: {
    type: Array,
    required: true,
  },
  // Chart height in pixels
  height: {
    type: Number,
    default: 300,
  },
  valueUnit: {
    type: String,
    default: 'ms',
  },
  yAxisLabel: {
    type: String,
    default: '',
  },
  // Show as relative speedup instead of absolute latency
  showSpeedup: {
    type: Boolean,
    default: false,
  },
})

const { isDark } = useData()
const chartRef = ref(null)
let chartInstance = null

const fontFamily = computed(() =>
  "'JetBrains Mono', ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"
)

function hexToRgba(color, alpha = 1) {
  const value = color.trim()

  if (!value) {
    return `rgba(46, 90, 232, ${alpha})`
  }

  if (value.startsWith('rgb')) {
    const channels = value.match(/[\d.]+/g)

    if (!channels || channels.length < 3) {
      return value
    }

    const [red, green, blue] = channels
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`
  }

  const normalized = value.replace('#', '')
  const expanded =
    normalized.length === 3
      ? normalized
          .split('')
          .map((channel) => `${channel}${channel}`)
          .join('')
      : normalized

  if (expanded.length !== 6) {
    return value
  }

  const red = Number.parseInt(expanded.slice(0, 2), 16)
  const green = Number.parseInt(expanded.slice(2, 4), 16)
  const blue = Number.parseInt(expanded.slice(4, 6), 16)

  return `rgba(${red}, ${green}, ${blue}, ${alpha})`
}

function resolveThemeValue(styles, token, fallback) {
  return styles.getPropertyValue(token).trim() || fallback
}

const colors = computed(() => {
  if (typeof window === 'undefined') {
    return {
      pytorch: 'rgba(46, 90, 232, 0.2)',
      pytorchBorder: '#2e5ae8',
      triton: 'rgba(13, 138, 134, 0.28)',
      tritonBorder: '#0d8a86',
      text: '#121b2d',
      textMuted: '#52607a',
      grid: 'rgba(24, 34, 53, 0.12)',
      axis: 'rgba(24, 34, 53, 0.2)',
      tooltipBg: '#ffffff',
      tooltipBorder: 'rgba(24, 34, 53, 0.12)',
      tooltipTitle: '#121b2d',
    }
  }

  const styles = getComputedStyle(document.documentElement)
  const accent = resolveThemeValue(styles, '--editorial-accent', '#2e5ae8')
  const signal = resolveThemeValue(styles, '--editorial-signal', '#0d8a86')
  const rule = resolveThemeValue(styles, '--editorial-rule', 'rgba(24, 34, 53, 0.12)')
  const ruleStrong = resolveThemeValue(styles, '--editorial-rule-strong', 'rgba(24, 34, 53, 0.2)')
  const text = resolveThemeValue(styles, '--vp-c-text-1', '#121b2d')
  const textMuted = resolveThemeValue(styles, '--vp-c-text-2', '#52607a')
  const tooltipBg = resolveThemeValue(styles, '--vp-c-bg-elv', '#ffffff')

  return {
    pytorch: hexToRgba(accent, isDark.value ? 0.28 : 0.16),
    pytorchBorder: accent,
    triton: hexToRgba(signal, isDark.value ? 0.4 : 0.24),
    tritonBorder: signal,
    text,
    textMuted,
    grid: rule,
    axis: ruleStrong,
    tooltipBg,
    tooltipBorder: ruleStrong,
    tooltipTitle: text,
  }
})

function createChart() {
  if (!chartRef.value) return

  const ctx = chartRef.value.getContext('2d')

  const datasets = props.showSpeedup
    ? [
        {
          label: 'Speedup (PyTorch → Triton)',
          data: props.data.map((d) => Number((d.pytorch / d.triton).toFixed(2))),
          backgroundColor: colors.value.triton,
          borderColor: colors.value.tritonBorder,
          borderWidth: 1.25,
          borderRadius: 10,
          borderSkipped: false,
          barPercentage: 0.6,
        },
      ]
    : [
        {
          label: 'PyTorch (baseline)',
          data: props.data.map((d) => d.pytorch),
          backgroundColor: colors.value.pytorch,
          borderColor: colors.value.pytorchBorder,
          borderWidth: 1.25,
          borderRadius: 10,
          borderSkipped: false,
          barPercentage: 0.8,
        },
        {
          label: 'Triton (optimized)',
          data: props.data.map((d) => d.triton),
          backgroundColor: colors.value.triton,
          borderColor: colors.value.tritonBorder,
          borderWidth: 1.25,
          borderRadius: 10,
          borderSkipped: false,
          barPercentage: 0.8,
        },
      ]

  chartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: props.data.map((d) => d.label),
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: !props.showSpeedup,
          position: 'top',
          align: 'start',
          labels: {
            boxWidth: 12,
            boxHeight: 12,
            usePointStyle: true,
            pointStyle: 'rectRounded',
            font: { family: fontFamily.value, size: 12 },
            color: colors.value.textMuted,
          },
        },
        tooltip: {
          backgroundColor: colors.value.tooltipBg,
          titleFont: { family: fontFamily.value, size: 13 },
          bodyFont: { family: fontFamily.value, size: 12 },
          titleColor: colors.value.tooltipTitle,
          bodyColor: colors.value.text,
          borderColor: colors.value.tooltipBorder,
          borderWidth: 1,
          padding: 12,
          displayColors: !props.showSpeedup,
          callbacks: {
            label: (context) => {
              if (props.showSpeedup) {
                return `${context.raw}× faster`
              }
              const value = context.raw
              const baseline = props.data[context.dataIndex].pytorch
              const speedup = (baseline / value).toFixed(2)
              return `${context.dataset.label}: ${value.toFixed(2)}${props.valueUnit} (${speedup}×)`
            },
          },
        },
      },
      scales: {
        x: {
          border: {
            color: colors.value.axis,
          },
          grid: { display: false },
          ticks: {
            font: { family: fontFamily.value, size: 12 },
            color: colors.value.textMuted,
          },
        },
        y: {
          beginAtZero: true,
          border: {
            color: colors.value.axis,
          },
          grid: { color: colors.value.grid },
          ticks: {
            font: { family: fontFamily.value, size: 12 },
            color: colors.value.textMuted,
          },
          title: {
            display: true,
            text:
              props.yAxisLabel ||
              (props.showSpeedup ? 'Speedup Factor' : `Latency (${props.valueUnit})`),
            font: { family: fontFamily.value, size: 13, weight: 500 },
            color: colors.value.text,
          },
        },
      },
    },
  })
}

function destroyChart() {
  if (chartInstance) {
    chartInstance.destroy()
    chartInstance = null
  }
}

onMounted(() => {
  createChart()
})

onUnmounted(() => {
  destroyChart()
})

watch(
  () => [props.data, isDark.value, props.showSpeedup, props.height, props.valueUnit, props.yAxisLabel],
  () => {
    destroyChart()
    createChart()
  },
  { deep: true }
)
</script>

<style scoped>
.perf-chart-container {
  height: v-bind('height + "px"');
  margin: 24px 0;
  position: relative;
}
</style>
