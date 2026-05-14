import DefaultTheme from 'vitepress/theme'
import './style.css'

// Import custom components
import HomeHero from './components/HomeHero.vue'
import KernelShowcase from './components/KernelShowcase.vue'
import ArchitecturePreview from './components/ArchitecturePreview.vue'
import PerformanceChart from './components/PerformanceChart.vue'

// Register components globally
export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('HomeHero', HomeHero)
    app.component('KernelShowcase', KernelShowcase)
    app.component('ArchitecturePreview', ArchitecturePreview)
    app.component('PerformanceChart', PerformanceChart)
  },
}
