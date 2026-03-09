---
title: "2026-03-10 — GitHub Pages 完善（第二轮）"
---

# 2026-03-10 — GitHub Pages 完善（第二轮）

## CI 工作流修复 (`ci.yml`)

1. **YAML 语法修复** — 4 处 `cache: pip` 缩进错误（与 `with:` 同级而非嵌套），导致 GitHub Actions 解析失败或忽略缓存
2. **移除 Black 依赖** — 项目已使用 Ruff，无需同时安装 Black；`Run Black` 步骤替换为 `Run Ruff format check`
3. **移除无用的注释掉的 GPU test job** — 减少文件噪音

## Pages 工作流优化 (`pages.yml`)

4. **`cancel-in-progress`** — `false` → `true`，避免过时提交阻塞新部署

## Jekyll 页面 front matter 补全

5. **`CHANGELOG.md`** — 添加 `layout: default` + `title` + `description` + 返回首页链接
6. **`CONTRIBUTING.md`** — 添加 `layout: default` + `title` + `description` + 返回首页链接
7. **`CODE_OF_CONDUCT.md`** — 添加 `layout: default` + `title` + `description` + 返回首页链接

## 主页增强 (`index.md`)

8. **添加 CI + Pages 徽章** — 与 README 保持一致

## 其他

9. **`.gitignore`** — 添加 `_site/`、`.jekyll-cache/`、`.jekyll-metadata`（Jekyll 构建产物）
10. **`changelog/index.md`** — 添加本轮变更条目
