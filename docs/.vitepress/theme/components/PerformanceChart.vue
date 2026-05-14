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

const colors = computed(() => ({
  pytorch: isDark.value ? 'rgba(52, 118, 246, 0.7)' : 'rgba(52, 118, 246, 0.8)',
  pytorchBorder: 'rgb(52, 118, 246)',
  triton: isDark.value ? 'rgba(118, 185, 0, 0.7)' : 'rgba(118, 185, 0, 0.8)',
  tritonBorder: 'rgb(118, 185, 0)',
  text: isDark.value ? '#c9d1d9' : '#24292f',
  grid: isDark.value ? 'rgba(139, 148, 158, 0.2)' : 'rgba(139, 148, 158, 0.3)',
}))

function createChart() {
  if (!chartRef.value) return

  const ctx = chartRef.value.getContext('2d')

  const datasets = props.showSpeedup
    ? [
        {
          label: 'Speedup (PyTorch → Triton)',
          data: props.data.map((d) => (d.pytorch / d.triton).toFixed(2)),
          backgroundColor: colors.value.triton,
          borderColor: colors.value.tritonBorder,
          borderWidth: 1,
          barPercentage: 0.6,
        },
      ]
    : [
        {
          label: 'PyTorch (baseline)',
          data: props.data.map((d) => d.pytorch),
          backgroundColor: colors.value.pytorch,
          borderColor: colors.value.pytorchBorder,
          borderWidth: 1,
          barPercentage: 0.8,
        },
        {
          label: 'Triton (optimized)',
          data: props.data.map((d) => d.triton),
          backgroundColor: colors.value.triton,
          borderColor: colors.value.tritonBorder,
          borderWidth: 1,
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
          labels: {
            font: { family: fontFamily.value, size: 12 },
            color: colors.value.text,
          },
        },
        tooltip: {
          backgroundColor: isDark.value ? '#21262d' : '#ffffff',
          titleFont: { family: fontFamily.value, size: 13 },
          bodyFont: { family: fontFamily.value, size: 12 },
          titleColor: colors.value.text,
          bodyColor: colors.value.text,
          borderColor: isDark.value ? '#30363d' : '#d0d7de',
          borderWidth: 1,
          padding: 12,
          callbacks: {
            label: (context) => {
              if (props.showSpeedup) {
                return `${context.raw}× faster`
              }
              const value = context.raw
              const baseline = props.data[context.dataIndex].pytorch
              const speedup = (baseline / value).toFixed(2)
              return `${context.dataset.label}: ${value.toFixed(2)}ms (${speedup}×)`
            },
          },
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: {
            font: { family: fontFamily.value, size: 12 },
            color: colors.value.text,
          },
        },
        y: {
          beginAtZero: true,
          grid: { color: colors.value.grid },
          ticks: {
            font: { family: fontFamily.value, size: 12 },
            color: colors.value.text,
          },
          title: {
            display: true,
            text: props.showSpeedup ? 'Speedup Factor' : 'Latency (ms)',
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
  () => [props.data, isDark.value],
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