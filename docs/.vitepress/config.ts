import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import llmstxt from 'vitepress-plugin-llms'

const rawBase = process.env.VITEPRESS_BASE
const base = rawBase
  ? rawBase.startsWith('/')
    ? rawBase.endsWith('/') ? rawBase : `${rawBase}/`
    : `/${rawBase}/`
  : '/'

const enSections = [
  { text: 'Overview', link: '/en/overview/', activeMatch: '/en/overview/', sidebarPrefix: '/en/overview/', sidebarLink: '/en/overview/' },
  { text: 'Academy', link: '/en/academy/', activeMatch: '/en/academy/', sidebarPrefix: '/en/academy/', sidebarLink: '/en/academy/' },
  { text: 'Kernel Families', link: '/en/kernel-families/', activeMatch: '/en/kernel-families/', sidebarPrefix: '/en/kernel-families/', sidebarLink: '/en/kernel-families/' },
  { text: 'Architecture Lab', link: '/en/architecture-lab/', activeMatch: '/en/architecture-lab/', sidebarPrefix: '/en/architecture-lab/', sidebarLink: '/en/architecture-lab/' },
  { text: 'Guides', link: '/en/guides/', activeMatch: '/en/guides/', sidebarPrefix: '/en/guides/', sidebarLink: '/en/guides/' },
  { text: 'Research', link: '/en/reference-research/', activeMatch: '/en/reference-research/', sidebarPrefix: '/en/reference-research/', sidebarLink: '/en/reference-research/' },
  { text: 'Release Notes', link: '/en/release-notes/changelog', activeMatch: '/en/release-notes/', sidebarPrefix: '/en/release-notes/', sidebarLink: '/en/release-notes/changelog', sidebarText: 'Changelog' },
] as const

const zhSections = [
  { text: '导读', link: '/zh/overview/', activeMatch: '/zh/overview/', sidebarPrefix: '/zh/overview/', sidebarLink: '/zh/overview/' },
  { text: '学院', link: '/zh/academy/', activeMatch: '/zh/academy/', sidebarPrefix: '/zh/academy/', sidebarLink: '/zh/academy/' },
  { text: '算子族', link: '/zh/kernel-families/', activeMatch: '/zh/kernel-families/', sidebarPrefix: '/zh/kernel-families/', sidebarLink: '/zh/kernel-families/' },
  { text: '架构实验室', link: '/zh/architecture-lab/', activeMatch: '/zh/architecture-lab/', sidebarPrefix: '/zh/architecture-lab/', sidebarLink: '/zh/architecture-lab/' },
  { text: '工程指南', link: '/zh/guides/', activeMatch: '/zh/guides/', sidebarPrefix: '/zh/guides/', sidebarLink: '/zh/guides/' },
  { text: '参考与研究', link: '/zh/reference-research/', activeMatch: '/zh/reference-research/', sidebarPrefix: '/zh/reference-research/', sidebarLink: '/zh/reference-research/' },
  { text: '发布说明', link: '/zh/release-notes/changelog', activeMatch: '/zh/release-notes/', sidebarPrefix: '/zh/release-notes/', sidebarLink: '/zh/release-notes/changelog', sidebarText: '变更日志' },
] as const

function buildNav(
  sections: ReadonlyArray<{ text: string; link: string; activeMatch: string }>,
) {
  return sections.map(({ text, link, activeMatch }) => ({ text, link, activeMatch }))
}

function buildSidebar(
  sections: ReadonlyArray<{
    text: string
    sidebarPrefix: string
    sidebarLink: string
    sidebarText?: string
  }>,
) {
  return Object.fromEntries(
    sections.map(({ text, sidebarPrefix, sidebarLink, sidebarText }) => [
      sidebarPrefix,
      [
        {
          text,
          items: [
            {
              text: sidebarText ?? text,
              link: sidebarLink,
            },
          ],
        },
      ],
    ]),
  )
}

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
        nav: buildNav(enSections),
        sidebar: buildSidebar(enSections),
      },
    },
    zh: {
      label: '简体中文',
      lang: 'zh-CN',
      link: '/zh/',
      title: 'Triton Fused Ops',
      description: '面向 Transformer 推理的高性能 Triton 融合算子库',
      themeConfig: {
        nav: buildNav(zhSections),
        sidebar: buildSidebar(zhSections),
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
