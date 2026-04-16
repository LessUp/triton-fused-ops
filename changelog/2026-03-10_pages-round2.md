---
layout: default
title: "2026-03-10 — GitHub Pages 完善（第二轮）"
description: "CI YAML 修复、front matter 补全、badges、.gitignore"
---

[← 返回更新日志](./)

# 2026-03-10 — GitHub Pages 完善（第二轮）

**日期:** 2026-03-10

---

## 🔧 CI 工作流修复 (`ci.yml`)

| 变更 | 说明 |
|------|------|
| **YAML 语法修复** | 4 处 `cache: pip` 缩进错误（与 `with:` 同级而非嵌套），导致 GitHub Actions 解析失败或忽略缓存 |
| **移除 Black 依赖** | 项目已使用 Ruff，无需同时安装 Black；`Run Black` 步骤替换为 `Run Ruff format check` |
| **移除无用注释** | 移除注释掉的 GPU test job，减少文件噪音 |

---

## ⚙️ Pages 工作流优化 (`pages.yml`)

| 变更 | 说明 |
|------|------|
| **cancel-in-progress** | `false` → `true`，避免过时提交阻塞新部署 |

---

## 📄 Jekyll 页面 front matter 补全

| 文件 | 添加内容 |
|------|----------|
| `CHANGELOG.md` | `layout: default` + `title` + `description` + 返回首页链接 |
| `CONTRIBUTING.md` | `layout: default` + `title` + `description` + 返回首页链接 |
| `CODE_OF_CONDUCT.md` | `layout: default` + `title` + `description` + 返回首页链接 |

---

## 🏠 主页增强 (`index.md`)

| 变更 | 说明 |
|------|------|
| **CI + Pages 徽章** | 与 README 保持一致 |

---

## 🙈 其他

| 变更 | 说明 |
|------|------|
| **`.gitignore`** | 添加 `_site/`、`.jekyll-cache/`、`.jekyll-metadata`（Jekyll 构建产物） |

---

## 变更文件

| 文件 | 变更类型 |
|------|----------|
| `.github/workflows/ci.yml` | Bug 修复 |
| `.github/workflows/pages.yml` | 优化 |
| `CHANGELOG.md` | front matter |
| `CONTRIBUTING.md` | front matter |
| `CODE_OF_CONDUCT.md` | front matter |
| `index.md` | 徽章 |
| `.gitignore` | Jekyll 产物 |
