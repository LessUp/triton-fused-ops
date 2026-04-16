---
layout: default
title: "2026-03-10 — GitHub Pages 优化"
description: "SEO 元数据、kramdown GFM、sparse checkout、changelog 索引"
---

[← 返回更新日志](./)

# 2026-03-10 — GitHub Pages 优化

**日期:** 2026-03-10

---

## 📝 Jekyll 配置增强 (`_config.yml`)

| 变更 | 说明 |
|------|------|
| **SEO 元数据** | 添加 `url`/`baseurl`/`lang`/`author`，改善搜索引擎索引 |
| **jekyll-seo-tag 插件** | 自动注入 Open Graph / Twitter Card 元数据，改善社交分享预览 |
| **kramdown GFM 设置** | 明确 `markdown: kramdown` + `input: GFM` + `syntax_highlighter: rouge` |
| **defaults 配置** | 全局默认 `layout: default`，changelog 目录也使用 default 布局 |
| **exclude 列表** | 排除源代码（`triton_ops`/`tests`/`examples`）、构建产物、配置文件 |

---

## ⚙️ 工作流优化 (`pages.yml`)

| 变更 | 说明 |
|------|------|
| **sparse checkout** | 仅检出文档相关文件，跳过 Python 源代码和测试 |
| **cone-mode: false** | 修复 cone 模式下文件级匹配问题 |
| **paths 触发条件** | 从通配 `'*.md'` 改为显式列出文档文件，避免源代码变更触发构建 |
| **job names** | `Build Pages` / `Deploy Pages` 更清晰的任务名称 |

---

## 📚 Changelog 可浏览化

| 变更 | 说明 |
|------|------|
| **新建 `changelog/index.md`** | 更新日志索引页，按时间倒序排列 |
| **YAML frontmatter** | 所有 changelog `.md` 添加 frontmatter，使 Jekyll 正确渲染 |

---

## 🏠 主页改进 (`index.md`)

| 变更 | 说明 |
|------|------|
| **SEO 描述** | frontmatter `description` 字段 |
| **"最近更新" section** | 展示最近变更摘要 + 链接到完整更新日志 |
| **链接优化** | 更新日志链接指向可浏览的 `changelog/` 索引 |

---

## 📖 README 增强

| 变更 | 说明 |
|------|------|
| **徽章** | `README.md` + `README.zh-CN.md` 添加 CI/Pages 徽章 |
| **项目主页链接** | 语言切换行增加项目主页入口 |

---

## 🔧 修复

| 变更 | 说明 |
|------|------|
| **CHANGELOG.md 比较链接** | `username/triton-fused-ops` → `LessUp/triton-fused-ops` |

---

## 变更文件

| 文件 | 变更类型 |
|------|----------|
| `_config.yml` | 配置增强 |
| `.github/workflows/pages.yml` | 工作流优化 |
| `changelog/index.md` | 新增 |
| `changelog/*.md` | frontmatter 添加 |
| `index.md` | 内容增强 |
| `README.md` | 徽章添加 |
| `README.zh-CN.md` | 徽章添加 |
