---
layout: default
title: "Triton Fused Ops Documentation"
description: "Complete documentation for Triton Fused Ops - High-performance GPU kernels for LLM inference"
---

# 📚 Triton Fused Ops Documentation

Welcome to the **Triton Fused Ops** documentation. This library provides high-performance GPU kernels for Transformer models with operator fusion and FP8 quantization.

---

## 🚀 Quick Start

New to Triton Fused Ops? Start here:

1. **[Installation](getting-started/installation.md)** — Install the library and dependencies
2. **[Quick Start](getting-started/quickstart.md)** — Get up and running in 5 minutes
3. **[Examples](getting-started/examples.md)** — Learn from practical code samples

---

## 📖 Documentation Sections

### Getting Started
Guides to help you get started quickly:

| Guide | Description |
|:------|:------------|
| [Installation](getting-started/installation.md) | System requirements and installation instructions |
| [Quick Start](getting-started/quickstart.md) | Your first fused kernel in 3 lines of code |
| [Examples](getting-started/examples.md) | Practical examples for common use cases |

### API Reference
Complete API documentation:

| Section | Description |
|:--------|:------------|
| [Core Kernels](api/kernels.md) | Fused RMSNorm+RoPE, Gated MLP, FP8 GEMM |
| [Quantization](api/quantization.md) | FP8 quantization utilities and best practices |
| [Auto-Tuning](api/autotuner.md) | Automatic kernel configuration optimization |
| [Benchmark](api/benchmark.md) | Performance measurement tools |

### User Guides
Detailed guides for specific topics:

| Guide | Description |
|:------|:------------|
| [Integration Guide](guides/integration.md) | Integrate with HuggingFace, PyTorch, vLLM |
| [Performance Tuning](guides/performance.md) | Optimize for your specific hardware |
| [FP8 Best Practices](guides/fp8-best-practices.md) | Get the most out of FP8 quantization |

### Internals
Technical deep dives:

| Document | Description |
|:---------|:------------|
| [Architecture](internals/architecture.md) | Overall library architecture |
| [Kernel Design](internals/kernel-design.md) | Triton kernel implementation details |
| [Memory Optimization](internals/memory-optimization.md) | Fusion strategies and memory optimization |

---

## ⚡ Performance Highlights

| Kernel | Speedup | Memory Savings |
|:-------|:-------:|:--------------:|
| `fused_rmsnorm_rope` | **~3x** | 50% fewer HBM writes |
| `fused_gated_mlp` | **~1.5x** | 1 intermediate tensor less |
| `fp8_gemm` | **~1.4x** | **50%** weight storage |

---

## 🔗 Quick Links

- [📖 README](../../README.md) — Project overview
- [🇨🇳 中文文档](../zh/) — Chinese documentation
- [📝 Changelog](../../CHANGELOG.md) — Version history
- [🤝 Contributing](../../CONTRIBUTING.md) — Contribution guidelines
- [💻 GitHub](https://github.com/LessUp/triton-fused-ops)

---

## 💬 Support

- **Issues:** [GitHub Issues](https://github.com/LessUp/triton-fused-ops/issues)
- **Discussions:** [GitHub Discussions](https://github.com/LessUp/triton-fused-ops/discussions)

---

<div align="center">

**[⬆ Back to Top](#-triton-fused-ops-documentation)**

</div>
