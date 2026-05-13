---
title: API Reference
description: "Public API reference for kernels, quantization, autotuning, validation, and support types"
---

# API Reference

This section documents the maintained API surface and adjacent support modules used by the repository.

## Root-package exports

The root package exports the public user-facing surface from `triton_ops.__init__`.

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

## Knowledge areas

<div class="link-grid link-grid-3">
  <a class="info-card" href="/en/api/kernels">
    <span class="card-kicker">Kernels</span>
    <strong>Core compute paths</strong>
    <span>Fused RMSNorm + RoPE, fused Gated MLP, FP8 GEMM, and module wrappers.</span>
  </a>
  <a class="info-card" href="/en/api/quantization">
    <span class="card-kicker">Quantization</span>
    <strong>FP8 storage and scaling</strong>
    <span>Round-trip helpers, scale semantics, and the overflow-handling helper path.</span>
  </a>
  <a class="info-card" href="/en/api/autotuner">
    <span class="card-kicker">Autotuning</span>
    <strong>Search, cache, and metrics</strong>
    <span>`TritonAutoTuner`, `ConfigCache`, config spaces, and performance metrics.</span>
  </a>
  <a class="info-card" href="/en/api/benchmark">
    <span class="card-kicker">Benchmark</span>
    <strong>Verification and reports</strong>
    <span>`BenchmarkSuite`, `CorrectnessVerifier`, report objects, and benchmark helpers.</span>
  </a>
  <a class="info-card" href="/en/api/models">
    <span class="card-kicker">Models</span>
    <strong>Dataclasses and result containers</strong>
    <span>`TensorSpec`, input specs, `KernelMetrics`, `TuningResult`, and `FP8Format`.</span>
  </a>
  <a class="info-card" href="/en/api/validation">
    <span class="card-kicker">Validation</span>
    <strong>Input checks and constraints</strong>
    <span>Shape, dtype, contiguity, device, and scalar-parameter validation helpers.</span>
  </a>
  <a class="info-card" href="/en/api/errors">
    <span class="card-kicker">Errors</span>
    <strong>Exception hierarchy</strong>
    <span>Device, dtype, shape, tuning, and overflow failure types with attached metadata.</span>
  </a>
</div>

## Important scope note

Some helper functions live in submodules without being exported at the root package. The API pages call out those import paths explicitly when relevant.
