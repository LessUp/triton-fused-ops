---
layout: default
title: "更新日志 — Triton Fused Ops"
description: "Triton Fused Ops 项目版本历史与变更记录"
---

[← 返回首页]({{ site.baseurl }}/)

# 更新日志

Triton Fused Ops 项目的版本历史与变更记录，按时间倒序排列。

---

## 版本索引

| 版本 | 日期 | 变更摘要 |
|------|------|----------|
| [**Unreleased**](#unreleased) | — | 最新开发进度 |
| [**v0.2.0**](#v020---2026-03-09) | 2026-03-09 | 重大重构 — SwiGLU 正确性修复、FP8Linear 优化 |
| [**v0.1.0**](#v010---2024-01-01) | 2024-01-01 | 初始发布 |

---

## 详细变更记录

### Unreleased

#### Added
- `DeviceError` 异常类 — 当 CUDA 不可用时提供更清晰的错误信息
- FP8 GEMM 自适应块大小启发式算法 — 根据矩阵维度自动选择最优块大小
- `ConfigCache` 日志记录 — 便于调试缓存操作

#### Changed
- 完善 `__init__.py` 模块级文档 — 包含快速开始示例、性能特性表、硬件要求
- 改进 FP8 量化内核文档 — 添加硬件兼容性说明

#### Fixed
- 修复 `autotuner/tuner.py` 占位符指标 — 添加正确的文档说明
- 修复缓存写入失败静默问题 — 现在会记录警告日志
- 所有内核函数添加 CUDA 可用性检查 — 提供清晰的错误信息

---

### v0.2.0 - 2026-03-09

[查看详细日志](2026-03-09_major-refactoring)

#### 🐛 关键 Bug 修复

**Gated MLP: gate/up 投影激活顺序错误（正确性问题）**
- 标准 SwiGLU 公式为 `output = activation(gate_proj(x)) * up_proj(x)`
- 之前的内核和参考实现错误地将激活函数应用于 `up_acc` 而非 `gate_acc`
- 这会导致所有使用 SwiGLU 的模型（LLaMA、Mistral 等）输出错误
- 已修复 Triton 内核 (`fused_gated_mlp_kernel`) 和 PyTorch 参考 (`gated_mlp_reference`)

**RMSNorm 内核: batch_idx 计算错误**
- `rmsnorm_kernel` 计算 `batch_idx = row_idx // cdiv(hidden_dim, BLOCK_SIZE)` 数学上错误
- 该变量未被使用，但表明对程序网格的理解有误
- 已移除错误的计算；`row_idx` 已经正确索引扁平化的 batch*seq 网格

#### ⚡ 性能优化

**FP8Linear: 权重转置缓存**
- `FP8Linear.forward()` 每次前向都调用 `self.weight_fp8.t().contiguous()` — 对大权重矩阵是昂贵的分配+拷贝
- 现在在 `quantize_weights()` 时预计算并缓存转置后的权重为 `weight_fp8_t`
- 消除每次前向传递的一次 GPU 分配

#### 🧹 代码质量

**api.py: 合并重复导入**
- 每个内核模块之前被两个独立的 `from...import` 块导入
- 已合并为每个模块单一导入语句

#### 📦 版本变更
- `0.1.0` → `0.2.0` (pyproject.toml + `__init__.py`)

---

### v0.1.0 - 2024-01-01

初始发布版本。

#### ✨ 新增功能

**融合内核**
- **RMSNorm + RoPE 融合** — 单内核完成归一化 + 旋转位置编码
- **Gated MLP 融合** — 单 pass 完成门控投影 + 激活函数（SiLU/GELU）
- **FP8 GEMM** — FP8 量化矩阵乘法，FP32 累加保证数值稳定
- **FP8 量化工具** — 动态范围计算、Scale Factor 计算、溢出处理

**基础设施**
- **Auto-Tuning 框架** — 自动搜索最优内核配置
- **Benchmark 套件** — 性能测量和正确性验证
- **完整测试套件** — 单元测试 + Hypothesis 属性测试

**文档**
- README 双语支持（英文/中文）
- API 文档和示例
- 贡献指南和行为准则

---

## 基础设施变更

以下变更不涉及版本号更新，但影响项目结构：

| 日期 | 变更 |
|------|------|
| [2026-03-10](2026-03-10_pages-round2) | Pages 完善 — CI YAML 修复、front matter 补全、badges |
| [2026-03-10](2026-03-10_pages-optimization) | GitHub Pages 优化 — SEO、kramdown GFM、sparse checkout |
| [2026-03-10](2026-03-10_workflow-deep-standardization) | Workflow 深度标准化 — 权限、并发、路径过滤 |
| [2025-02-13](2025-02-13_project-infrastructure) | 项目基础设施优化 — LICENSE、.gitignore、.editorconfig |

---

## 版本链接

- [Unreleased]: https://github.com/LessUp/triton-fused-ops/compare/v0.2.0...HEAD
- [v0.2.0]: https://github.com/LessUp/triton-fused-ops/compare/v0.1.0...v0.2.0
- [v0.1.0]: https://github.com/LessUp/triton-fused-ops/releases/tag/v0.1.0
