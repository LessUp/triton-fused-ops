---
layout: page
title: Triton Fused Ops
description: High-performance Triton kernels for Transformer inference
---

<script setup>
import HomeHero from '@theme/components/HomeHero.vue'
import KernelShowcase from '@theme/components/KernelShowcase.vue'
import ArchitecturePreview from '@theme/components/ArchitecturePreview.vue'
</script>

<HomeHero />

## Core Kernels

<KernelShowcase />

## Architecture at a Glance

<ArchitecturePreview />

## Why Triton Fused Ops?

| Feature | Benefit |
|---------|---------|
| **Kernel Fusion** | Single kernel launch for RMSNorm + RoPE eliminates intermediate HBM round-trips |
| **CPU References** | Pure NumPy implementations enable correctness verification without GPU |
| **Auto-Tuning** | TritonAutoTuner discovers optimal launch parameters per hardware automatically |
| **FP8 Support** | E4M3/E5M2 quantized GEMM reduces weight memory by 50% |

## Quick Start

```bash
pip install triton-fused-ops
```

```python
import torch
from triton_ops import fused_rmsnorm_rope

x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
weight = torch.ones(4096, device="cuda", dtype=torch.float16)
cos = torch.randn(128, 64, device="cuda", dtype=torch.float16)
sin = torch.randn(128, 64, device="cuda", dtype=torch.float16)

y = fused_rmsnorm_rope(x, weight, cos, sin)
```

## Next Steps

- [Installation Guide](/en/getting-started/installation) — Detailed setup instructions
- [API Reference](/en/api/kernels) — Kernel documentation
- [Architecture](/en/internals/architecture) — How it works internally
- [Performance Guide](/en/guides/performance) — Optimization tips
