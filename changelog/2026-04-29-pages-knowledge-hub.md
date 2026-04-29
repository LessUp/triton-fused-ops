# 2026-04-29 Pages Knowledge Hub

## 中文

### 变更目标

- 将 GitHub Pages 从“仓库附属页”重构为只保留知识内容的双语技术知识库。
- 清理更新日志、贡献说明、仓库流程文件等不应发布到 Pages 的内容入口。
- 让文档说明与当前仓库真实代码行为保持一致，减少 API 与示例漂移。

### 本次修改

- 重构根首页 `index.md` 与 `docs/index.md`，改为知识库入口与双语语言入口。
- 为中英文文档分别建立四组知识结构：`Getting Started` / `开始使用`、`API`、`Guides` / `工程指南`、`Internals` / `内部实现`。
- 新增 API 参考页：`models`、`validation`、`errors`，补齐原先缺失的知识点。
- 重写核心 API、量化、自动调优、benchmark、集成、性能、内部实现等页面内容，使其与 `triton_ops/` 当前实现和测试行为对齐。
- 将 `README`、`CHANGELOG`、`CONTRIBUTING`、`AGENTS`、`CLAUDE`、`changelog/` 等内容排除出 Pages 发布范围。
- 接入 `_includes/head_custom.html` 与 `_includes/footer_custom.html`，让自定义 CSS/JS 真正参与站点渲染。
- 重写 `assets/css/custom.scss` 与 `assets/js/custom.js`，新增知识首页视觉层、代码复制、页内目录、双语切换入口与更清晰的阅读层级。
- 简化 `robots.txt`，移除手写 `sitemap.xml`，交由 `jekyll-sitemap` 生成。
- 新建 OpenSpec change：`openspec/changes/optimize-pages-knowledge-hub/`。

### 验证情况

- 已完成静态结构检查与链接/内容残留清理。
- 未能完成本地 Jekyll 构建验证：当前环境缺少 `ruby`、`gem`、`bundle`，也没有 `docker` 可作为替代构建环境。

## English

### Purpose

- Reframe GitHub Pages into a bilingual technical knowledge hub.
- Remove non-knowledge repository surfaces from the published site.
- Align the published documentation with the repository's current code and tests.

### Verification

- Content and navigation were restructured and statically checked.
- Full local Jekyll build verification was blocked because the environment does not provide Ruby/Bundler or Docker.
