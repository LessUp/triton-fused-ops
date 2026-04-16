---
layout: default
title: "API 参考 — Triton Fused Ops"
description: "Triton Fused Ops 完整 API 参考 - 核心算子、量化、自动调优和基准测试工具"
---

# 📖 API 参考

Triton Fused Ops 的完整 API 文档。

---

## 📚 API 章节

| 章节 | 说明 | 链接 |
|:--------|:------------|:-----|
| **核心算子** | 融合 RMSNorm+RoPE、Gated MLP、FP8 GEMM | [kernels.md](kernels.md) |
| **量化** | FP8 量化工具 | [quantization.md](quantization.md) |
| **自动调优** | 自动算子配置 | [autotuner.md](autotuner.md) |
| **基准测试** | 性能测量工具 | [benchmark.md](benchmark.md) |

---

## 🎯 API 快速概览

### 函数式 API

```python
from triton_ops import (
    # 融合算子
    fused_rmsnorm_rope,      # RMSNorm + RoPE 融合
    fused_gated_mlp,         # Gated MLP (SwiGLU/GeGLU)
    fp8_gemm,                # FP8 量化 GEMM
    
    # 量化
    quantize_fp8,            # 量化为 FP8
    dequantize_fp8,          # 从 FP8 反量化
)
```

### Module API

```python
from triton_ops import (
    # PyTorch 模块
    FusedRMSNormRoPE,        # RMSNorm + RoPE 模块
    FusedGatedMLP,           # Gated MLP 模块
    FP8Linear,               # FP8 量化线性层
    
    # 自动调优
    TritonAutoTuner,         # 自动调优框架
    ConfigCache,             # 配置缓存
)
```

---

## 🔗 导航

### 核心算子
- [`fused_rmsnorm_rope`](kernels.md#fused_rmsnorm_rope) — 融合 RMSNorm + RoPE
- [`fused_gated_mlp`](kernels.md#fused_gated_mlp) — 融合 Gated MLP
- [`fp8_gemm`](kernels.md#fp8_gemm) — FP8 量化 GEMM
- [`FusedRMSNormRoPE`](kernels.md#fusedrmsnormrope) — 模块封装
- [`FusedGatedMLP`](kernels.md#fusedgatedmlp) — 模块封装
- [`FP8Linear`](kernels.md#fp8linear) — 量化线性层

### 量化
- [`quantize_fp8`](quantization.md#quantize_fp8) — 量化为 FP8
- [`dequantize_fp8`](quantization.md#dequantize_fp8) — 从 FP8 反量化
- [`FP8Format`](quantization.md#fp8format) — FP8 格式工具

### 自动调优
- [`TritonAutoTuner`](autotuner.md#tritonautotuner) — 配置搜索
- [`ConfigCache`](autotuner.md#configcache) — 持久化缓存
- [`TuningResult`](autotuner.md#tuningresult) — 调优结果

---

## 🌐 其他语言

- [🇺🇸 English API Docs](../../en/api/)

---

<div align="center">

**[⬆ 返回顶部](#-api-参考)** | **[← 返回文档](../)**

</div>
