---
layout: default
title: "Triton Fused Ops 文档"
description: "Triton Fused Ops 完整中文文档 - 面向 Transformer 模型的高性能 GPU 算子库"
---

# 📚 Triton Fused Ops 中文文档

欢迎使用 **Triton Fused Ops** 官方中文文档。本库为 Transformer 模型提供高性能 GPU 算子，支持算子融合和 FP8 量化。

---

## 🚀 快速开始

初次使用 Triton Fused Ops？从这里开始：

1. **[安装指南](getting-started/installation.md)** — 安装库和依赖项
2. **[快速开始](getting-started/quickstart.md)** — 5 分钟上手
3. **[示例教程](getting-started/examples.md)** — 通过实用代码示例学习

---

## 📖 文档目录

### 入门指南
帮助您快速上手的教程：

| 指南 | 说明 |
|:------|:------------|
| [安装指南](getting-started/installation.md) | 系统要求和安装说明 |
| [快速开始](getting-started/quickstart.md) | 3 行代码运行您的第一个融合算子 |
| [示例教程](getting-started/examples.md) | 常见用例的实用示例 |

### API 参考
完整的 API 文档：

| 章节 | 说明 |
|:--------|:------------|
| [核心算子](api/kernels.md) | 融合 RMSNorm+RoPE、Gated MLP、FP8 GEMM |
| [量化](api/quantization.md) | FP8 量化工具和最佳实践 |
| [自动调优](api/autotuner.md) | 自动算子配置优化 |
| [基准测试](api/benchmark.md) | 性能测量工具 |

### 用户指南
特定主题的详细指南：

| 指南 | 说明 |
|:------|:------------|
| [集成指南](guides/integration.md) | 与 HuggingFace、PyTorch、vLLM 集成 |
| [性能优化](guides/performance.md) | 针对您的硬件进行优化 |
| [FP8 最佳实践](guides/fp8-best-practices.md) | 充分利用 FP8 量化 |

### 内部实现
技术深度解析：

| 文档 | 说明 |
|:---------|:------------|
| [架构设计](internals/architecture.md) | 整体库架构 |
| [Kernel 设计](internals/kernel-design.md) | Triton 内核实现细节 |
| [内存优化](internals/memory-optimization.md) | 融合策略和内存优化 |

---

## ⚡ 性能亮点

| 算子 | 加速比 | 内存节省 |
|:-------|:-------:|:--------------:|
| `fused_rmsnorm_rope` | **~3 倍** | HBM 写入减少 50% |
| `fused_gated_mlp` | **~1.5 倍** | 减少 1 个中间张量 |
| `fp8_gemm` | **~1.4 倍** | **50%** 权重存储 |

---

## 🔗 快速链接

- [📖 README](../../README.zh-CN.md) — 项目概览
- [🇺🇸 English Docs](../en/) — 英文文档
- [📝 更新日志](../../CHANGELOG.zh-CN.md) — 版本历史
- [🤝 贡献指南](../../CONTRIBUTING.md) — 参与贡献
- [💻 GitHub](https://github.com/LessUp/triton-fused-ops)

---

## 💬 获取帮助

- **Issues:** [GitHub Issues](https://github.com/LessUp/triton-fused-ops/issues)
- **讨论区:** [GitHub Discussions](https://github.com/LessUp/triton-fused-ops/discussions)

---

<div align="center">

**[⬆ 返回顶部](#-triton-fused-ops-中文文档)**

</div>
