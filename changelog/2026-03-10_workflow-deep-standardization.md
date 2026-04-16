---
layout: default
title: "2026-03-10 — Workflow 深度标准化"
description: "CI workflow 统一权限、并发配置与路径过滤"
---

[← 返回更新日志](./)

# Workflow 深度标准化

**日期:** 2026-03-10

---

## 变更内容

| 变更 | 说明 |
|------|------|
| **CI workflow 权限** | 统一 `permissions: contents: read` |
| **并发配置** | 添加 `concurrency` 配置，避免重复运行 |
| **Pages workflow** | 补充 `actions/configure-pages@v5` 步骤 |
| **路径过滤** | Pages workflow 添加 `paths` 触发过滤，减少无效构建 |

---

## 背景

全仓库第二轮 GitHub Actions 深度标准化：统一命名、权限、并发、路径过滤与缓存策略。

---

## 变更文件

| 文件 | 变更类型 |
|------|----------|
| `.github/workflows/ci.yml` | 权限 + 并发 |
| `.github/workflows/pages.yml` | 配置步骤 + 路径过滤 |
