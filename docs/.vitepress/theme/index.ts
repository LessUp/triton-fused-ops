import DefaultTheme from 'vitepress/theme'
import './style.css'

import HomeHero from './components/HomeHero.vue'
import KernelShowcase from './components/KernelShowcase.vue'
import ArchitecturePreview from './components/ArchitecturePreview.vue'
import PerformanceChart from './components/PerformanceChart.vue'
import WhitepaperHero from './components/WhitepaperHero.vue'
import ReaderTracks from './components/ReaderTracks.vue'
import KernelAtlas from './components/KernelAtlas.vue'
import SystemBlueprint from './components/SystemBlueprint.vue'
import FigureFrame from './components/FigureFrame.vue'
import ResearchLandscape from './components/ResearchLandscape.vue'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('HomeHero', HomeHero)
    app.component('KernelShowcase', KernelShowcase)
    app.component('ArchitecturePreview', ArchitecturePreview)
    app.component('PerformanceChart', PerformanceChart)
    app.component('WhitepaperHero', WhitepaperHero)
    app.component('ReaderTracks', ReaderTracks)
    app.component('KernelAtlas', KernelAtlas)
    app.component('SystemBlueprint', SystemBlueprint)
    app.component('FigureFrame', FigureFrame)
    app.component('ResearchLandscape', ResearchLandscape)
  },
}
