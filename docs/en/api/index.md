---
layout: default
title: "API Reference — Triton Fused Ops"
description: "Complete API reference for Triton Fused Ops - kernels, quantization, autotuner, and benchmark tools"
---

# 📖 API Reference

Complete API documentation for Triton Fused Ops.

---

## 📚 API Sections

| Section | Description | Link |
|:--------|:------------|:-----|
| **Core Kernels** | Fused RMSNorm+RoPE, Gated MLP, FP8 GEMM | [kernels.md](kernels.md) |
| **Quantization** | FP8 quantization utilities | [quantization.md](quantization.md) |
| **Auto-Tuning** | Automatic kernel configuration | [autotuner.md](autotuner.md) |
| **Benchmark** | Performance measurement tools | [benchmark.md](benchmark.md) |

---

## 🎯 Quick API Overview

### Functional API

```python
from triton_ops import (
    # Fused kernels
    fused_rmsnorm_rope,      # RMSNorm + RoPE fusion
    fused_gated_mlp,         # Gated MLP (SwiGLU/GeGLU)
    fp8_gemm,                # FP8 quantized GEMM
    
    # Quantization
    quantize_fp8,            # Quantize to FP8
    dequantize_fp8,          # Dequantize from FP8
)
```

### Module API

```python
from triton_ops import (
    # PyTorch modules
    FusedRMSNormRoPE,        # Module for RMSNorm + RoPE
    FusedGatedMLP,           # Module for Gated MLP
    FP8Linear,               # FP8 quantized linear layer
    
    # Autotuning
    TritonAutoTuner,         # Auto-tuning framework
    ConfigCache,             # Configuration cache
)
```

---

## 🔗 Navigation

### Core Kernels
- [`fused_rmsnorm_rope`](kernels.md#fused_rmsnorm_rope) — Fused RMSNorm + RoPE
- [`fused_gated_mlp`](kernels.md#fused_gated_mlp) — Fused Gated MLP
- [`fp8_gemm`](kernels.md#fp8_gemm) — FP8 quantized GEMM
- [`FusedRMSNormRoPE`](kernels.md#fusedrmsnormrope) — Module wrapper
- [`FusedGatedMLP`](kernels.md#fusedgatedmlp) — Module wrapper
- [`FP8Linear`](kernels.md#fp8linear) — Quantized linear layer

### Quantization
- [`quantize_fp8`](quantization.md#quantize_fp8) — Quantize to FP8
- [`dequantize_fp8`](quantization.md#dequantize_fp8) — Dequantize from FP8
- [`FP8Format`](quantization.md#fp8format) — FP8 format utilities

### Auto-Tuning
- [`TritonAutoTuner`](autotuner.md#tritonautotuner) — Configuration search
- [`ConfigCache`](autotuner.md#configcache) — Persistent cache
- [`TuningResult`](autotuner.md#tuningresult) — Tuning results

---

## 🌐 Other Languages

- [🇨🇳 中文 API 文档](../../zh/api/)

---

<div align="center">

**[⬆ Back to Top](#-api-reference)** | **[← Back to Documentation](../)**

</div>
