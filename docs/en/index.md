---
layout: default
title: Documentation
nav_order: 2
has_children: true
permalink: /docs/en/
---

# Triton Fused Ops Documentation

Use this section for practical usage, API references, and integration notes.

## Start here

| Guide | Purpose |
|:--|:--|
| [Installation](getting-started/installation) | Environment requirements and setup |
| [Quick Start](getting-started/quickstart) | Minimal first run |
| [Examples](getting-started/examples) | End-to-end usage samples |

## API and internals

| Section | Purpose |
|:--|:--|
| [Core Kernels](api/kernels) | Fused RMSNorm+RoPE, Gated MLP, FP8 GEMM |
| [Quantization](api/quantization) | FP8 quantization behavior and caveats |
| [Auto-Tuning](api/autotuner) | Tuning strategy and cache behavior |
| [Benchmark](api/benchmark) | Benchmark interfaces and output |
| [Architecture](internals/architecture) | Library-level design |

## Runtime boundary reminder

- GPU is required for Triton kernel execution.
- CPU-only environments are suitable for repository baseline checks and non-kernel validation.

## Quick links

- [Project Home](../../)
- [Chinese Docs](../zh/)
- [Changelog](../../CHANGELOG)
- [Contributing](https://github.com/LessUp/triton-fused-ops/blob/main/CONTRIBUTING.md)
