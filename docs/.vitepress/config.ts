import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import llmstxt from 'vitepress-plugin-llms'

const rawBase = process.env.VITEPRESS_BASE
const base = rawBase
  ? rawBase.startsWith('/')
    ? rawBase.endsWith('/') ? rawBase : `${rawBase}/`
    : `/${rawBase}/`
  : '/'

export default withMermaid(defineConfig({
  base,
  cleanUrls: true,
  title: 'Triton Fused Ops',
  description: 'High-performance Triton kernels for Transformer inference (RMSNorm+RoPE, Gated MLP, FP8 GEMM)',

  vite: {
    plugins: [llmstxt()],
    build: {
      chunkSizeWarningLimit: 1000,
    },
  },

  locales: {
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      title: 'Triton Fused Ops',
      description: 'High-performance Triton kernels for Transformer inference',
      themeConfig: {
        nav: [
          { text: 'Getting Started', link: '/en/getting-started/', activeMatch: '/en/getting-started/' },
          { text: 'API', link: '/en/api/', activeMatch: '/en/api/' },
          { text: 'Guides', link: '/en/guides/', activeMatch: '/en/guides/' },
          { text: 'Internals', link: '/en/internals/', activeMatch: '/en/internals/' },
          { text: 'References', link: '/en/references/', activeMatch: '/en/references/' },
          { text: 'Changelog', link: '/en/release-notes/changelog', activeMatch: '/en/release-notes/' },
        ],
        sidebar: {
          '/en/getting-started/': [
            {
              text: 'Getting Started',
              items: [
                { text: 'Installation', link: '/en/getting-started/installation' },
                { text: 'Quick Start', link: '/en/getting-started/quickstart' },
                { text: 'Examples', link: '/en/getting-started/examples' },
              ],
            },
          ],
          '/en/api/': [
            {
              text: 'API Reference',
              items: [
                { text: 'Core Kernels', link: '/en/api/kernels' },
                { text: 'FP8 Quantization', link: '/en/api/quantization' },
                { text: 'Auto-Tuning', link: '/en/api/autotuner' },
                { text: 'Benchmarking', link: '/en/api/benchmark' },
                { text: 'Models and Types', link: '/en/api/models' },
                { text: 'Validation', link: '/en/api/validation' },
                { text: 'Errors', link: '/en/api/errors' },
              ],
            },
          ],
          '/en/guides/': [
            {
              text: 'Guides',
              items: [
                { text: 'Integration', link: '/en/guides/integration' },
                { text: 'Performance', link: '/en/guides/performance' },
                { text: 'FP8 Best Practices', link: '/en/guides/fp8-best-practices' },
                { text: 'Benchmark Visualization', link: '/en/guides/benchmark-visualization' },
              ],
            },
          ],
          '/en/internals/': [
            {
              text: 'Internals',
              items: [
                { text: 'Architecture', link: '/en/internals/architecture' },
                { text: 'Kernel Design', link: '/en/internals/kernel-design' },
                { text: 'Memory Optimization', link: '/en/internals/memory-optimization' },
              ],
            },
          ],
          '/en/references/': [
            {
              text: 'References',
              items: [
                { text: 'Papers', link: '/en/references/papers' },
                { text: 'Projects', link: '/en/references/projects' },
                { text: 'Blogs & Docs', link: '/en/references/blogs' },
              ],
            },
          ],
          '/en/release-notes/': [
            {
              text: 'Release Notes',
              items: [
                { text: 'Changelog', link: '/en/release-notes/changelog' },
              ],
            },
          ],
        },
      },
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      title: 'Triton Fused Ops',
      description: '面向 Transformer 推理的高性能 Triton 融合算子库',
      themeConfig: {
        nav: [
          { text: '开始使用', link: '/zh/getting-started/', activeMatch: '/zh/getting-started/' },
          { text: 'API', link: '/zh/api/', activeMatch: '/zh/api/' },
          { text: '工程指南', link: '/zh/guides/', activeMatch: '/zh/guides/' },
          { text: '内部实现', link: '/zh/internals/', activeMatch: '/zh/internals/' },
          { text: '参考文献', link: '/zh/references/', activeMatch: '/zh/references/' },
          { text: '变更日志', link: '/zh/release-notes/changelog', activeMatch: '/zh/release-notes/' },
        ],
        sidebar: {
          '/zh/getting-started/': [
            {
              text: '开始使用',
              items: [
                { text: '安装指南', link: '/zh/getting-started/installation' },
                { text: '快速开始', link: '/zh/getting-started/quickstart' },
                { text: '示例教程', link: '/zh/getting-started/examples' },
              ],
            },
          ],
          '/zh/api/': [
            {
              text: 'API 参考',
              items: [
                { text: '核心算子', link: '/zh/api/kernels' },
                { text: 'FP8 量化', link: '/zh/api/quantization' },
                { text: '自动调优', link: '/zh/api/autotuner' },
                { text: '基准测试', link: '/zh/api/benchmark' },
                { text: '数据模型与类型', link: '/zh/api/models' },
                { text: '输入校验', link: '/zh/api/validation' },
                { text: '异常模型', link: '/zh/api/errors' },
              ],
            },
          ],
          '/zh/guides/': [
            {
              text: '工程指南',
              items: [
                { text: '集成指南', link: '/zh/guides/integration' },
                { text: '性能优化', link: '/zh/guides/performance' },
                { text: 'FP8 最佳实践', link: '/zh/guides/fp8-best-practices' },
                { text: '性能可视化', link: '/zh/guides/benchmark-visualization' },
              ],
            },
          ],
          '/zh/internals/': [
            {
              text: '内部实现',
              items: [
                { text: '架构设计', link: '/zh/internals/architecture' },
                { text: '算子设计', link: '/zh/internals/kernel-design' },
                { text: '内存优化', link: '/zh/internals/memory-optimization' },
              ],
            },
          ],
          '/zh/references/': [
            {
              text: '参考文献',
              items: [
                { text: '论文', link: '/zh/references/papers' },
                { text: '项目', link: '/zh/references/projects' },
                { text: '博客与文档', link: '/zh/references/blogs' },
                { text: '术语翻译对照', link: '/zh/references/translations' },
              ],
            },
          ],
          '/zh/release-notes/': [
            {
              text: '变更日志',
              items: [
                { text: 'Changelog', link: '/zh/release-notes/changelog' },
              ],
            },
          ],
        },
      },
    },
  },

  themeConfig: {
    appearance: 'dark',
    outline: [2, 3],
    search: { provider: 'local' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/LessUp/triton-fused-ops' },
    ],
  },

  mermaid: {
    theme: 'dark',
    themeVariables: {
      primaryColor: '#1a2e1a',
      primaryTextColor: '#c9d1d9',
      primaryBorderColor: '#76B900',
      lineColor: '#8b949e',
      secondaryColor: '#161b22',
      tertiaryColor: '#21262d',
      fontFamily: 'JetBrains Mono, ui-monospace, monospace',
    },
    flowchart: {
      curve: 'basis',
      padding: 20,
    },
    sequence: {
      actorMargin: 50,
      boxMargin: 10,
    },
  },
}))
