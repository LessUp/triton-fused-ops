---
title: API 参考
description: "核心算子、量化、自动调优、校验与支撑类型的公开 API 参考"
---

# API 参考

本节聚焦仓库当前维护的 API 表面，以及与这些接口密切相关的支撑模块。

## 根包导出

根包 `triton_ops` 从 `__init__.py` 暴露主要用户接口。

```python
from triton_ops import (
    fused_rmsnorm_rope,
    fused_gated_mlp,
    fp8_gemm,
    quantize_fp8,
    dequantize_fp8,
    FusedRMSNormRoPE,
    FusedGatedMLP,
    FP8Linear,
    TritonAutoTuner,
    ConfigCache,
    BenchmarkSuite,
)
```

## 知识分区

<div class="link-grid link-grid-3">
  <a class="info-card" href="/zh/api/kernels">
    <span class="card-kicker">Kernels</span>
    <strong>核心计算路径</strong>
    <span>融合 RMSNorm + RoPE、融合 Gated MLP、FP8 GEMM 与模块封装。</span>
  </a>
  <a class="info-card" href="/zh/api/quantization">
    <span class="card-kicker">Quantization</span>
    <strong>FP8 存储与 scale 语义</strong>
    <span>说明量化/反量化、scale 规则，以及溢出处理 helper 的真实导入路径。</span>
  </a>
  <a class="info-card" href="/zh/api/autotuner">
    <span class="card-kicker">Autotuning</span>
    <strong>搜索、缓存与指标</strong>
    <span>涵盖 `TritonAutoTuner`、`ConfigCache`、配置空间与性能指标。</span>
  </a>
  <a class="info-card" href="/zh/api/benchmark">
    <span class="card-kicker">Benchmark</span>
    <strong>正确性验证与报告</strong>
    <span>涵盖 `BenchmarkSuite`、`CorrectnessVerifier` 以及报告对象。</span>
  </a>
  <a class="info-card" href="/zh/api/models">
    <span class="card-kicker">Models</span>
    <strong>数据模型与结果容器</strong>
    <span>介绍 `TensorSpec`、输入规格、`KernelMetrics`、`TuningResult` 与 `FP8Format`。</span>
  </a>
  <a class="info-card" href="/zh/api/validation">
    <span class="card-kicker">Validation</span>
    <strong>输入校验与运行契约</strong>
    <span>说明 shape、dtype、连续内存、device 和标量参数检查。</span>
  </a>
  <a class="info-card" href="/zh/api/errors">
    <span class="card-kicker">Errors</span>
    <strong>异常层级</strong>
    <span>说明设备、dtype、形状、调优与数值溢出错误模型。</span>
  </a>
</div>

## 范围提醒

有些 helper 位于子模块里，但没有从根包重新导出。相关 API 页会明确给出真实导入路径，避免误导。
