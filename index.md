---
layout: default
title: Home
nav_order: 1
head_title: "Triton Fused Ops — Curated landing page"
description: "Fused Triton kernels for Transformer inference: RMSNorm+RoPE, Gated MLP, and FP8 GEMM."
---

# Triton Fused Ops

High-performance Triton kernels for Transformer inference paths where memory movement dominates runtime.

[Get Started](docs/en/getting-started/quickstart/){: .btn .btn-primary .fs-5 .mr-3 }
[Documentation](docs/en/){: .btn .btn-green .fs-5 .mr-3 }
[GitHub Repository](https://github.com/LessUp/triton-fused-ops){: .btn .fs-5 }

---

## Why this project

The core optimization target is reducing redundant HBM traffic in common inference subgraphs.

| Kernel family | Intent | Typical observed speedup* |
|:--|:--|:--:|
| `fused_rmsnorm_rope` | Fuse RMSNorm + RoPE | up to ~3x |
| `fused_gated_mlp` | Fuse gate/up + activation | ~1.3x–1.8x |
| `fp8_gemm` | FP8 GEMM path with scaling | ~1.2x–1.5x |

\*Measured in repository benchmarks (A100/CUDA 12.1). Results vary by hardware and workload.

---

## What to expect

- **GPU required** for Triton kernel runtime and performance tests.
- **CPU-safe validation path** is available for lint/type/tests/build checks.
- **OpenSpec-driven development** for non-trivial changes.

---

## Where to go next

- [Installation](docs/en/getting-started/installation/)
- [Quick Start](docs/en/getting-started/quickstart/)
- [Examples](docs/en/getting-started/examples/)
- [API Reference](docs/en/api/kernels/)
- [Chinese Documentation](docs/zh/)

---

<p class="text-small text-grey-dk-100">
MIT licensed · Built with OpenAI Triton
</p>
