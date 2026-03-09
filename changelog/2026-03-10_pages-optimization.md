---
title: "2026-03-10 — GitHub Pages 优化"
---

# 2026-03-10 — GitHub Pages 优化

## Jekyll 配置增强 (`_config.yml`)

1. **添加 SEO 元数据** — `url`/`baseurl`/`lang`/`author`，改善搜索引擎索引
2. **添加 `jekyll-seo-tag` 插件** — 自动注入 Open Graph / Twitter Card 元数据，改善社交分享预览
3. **添加 kramdown GFM 设置** — 明确 `markdown: kramdown` + `input: GFM` + `syntax_highlighter: rouge`，改善表格和代码块渲染
4. **添加 `defaults`** — 全局默认 `layout: default`，changelog 目录也默认使用 default 布局
5. **添加 `exclude` 列表** — 排除源代码（`triton_ops`/`tests`/`examples`）、构建产物、配置文件等无关内容

## 工作流优化 (`pages.yml`)

6. **添加 sparse checkout** — 仅检出文档相关文件，跳过 Python 源代码和测试
7. **添加 `sparse-checkout-cone-mode: false`** — 修复 cone 模式下文件级匹配问题
8. **细化 `paths` 触发条件** — 从通配 `'*.md'` 改为显式列出文档文件，避免源代码 `.md` 变更触发构建
9. **添加 job names** — `Build Pages` / `Deploy Pages`

## Changelog 可浏览化

10. **新建 `changelog/index.md`** — 更新日志索引页，按时间倒序排列
11. **所有 changelog `.md` 添加 YAML frontmatter** — 使 Jekyll 将其渲染为正式页面

## 主页改进 (`index.md`)

12. **添加 SEO 描述** — frontmatter `description` 字段
13. **添加"最近更新"section** — 展示最近变更摘要 + "查看完整更新日志"链接
14. **链接优化** — 更新日志链接指向可浏览的 `changelog/` 索引

## README 增强

15. **`README.md` + `README.zh-CN.md` 添加 CI/Pages 徽章** — 链接到 GitHub Actions 和项目主页
16. **添加 Project Page 链接** — 语言切换行增加项目主页入口

## 修复

17. **`CHANGELOG.md` 修复比较链接** — `username/triton-fused-ops` → `LessUp/triton-fused-ops`，添加 v0.2.0 版本链接
