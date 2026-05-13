---
layout: home
---

<div class="home-header">
  <div class="home-header-left">
    <div class="home-logo">TFO</div>
    <div>
      <span class="home-title">Triton Fused Ops</span>
      <span class="home-subtitle">High-Performance GPU Kernels for Transformer Inference</span>
    </div>
  </div>
  <div class="home-nav">
    <a href="/en/getting-started/">Getting Started</a>
    <a href="https://github.com/LessUp/triton-fused-ops">GitHub</a>
    <a href="/zh/">中文</a>
  </div>
</div>

<div class="home-intro-row">
  <div class="home-intro">
    Fused GPU kernels for LLM inference. Memory-bound &rarr; Compute-bound. This library provides production-ready Triton implementations of RMSNorm+RoPE fusion, Gated MLP fusion, and FP8 GEMM with auto-tuning and benchmarking infrastructure.
  </div>
  <div class="home-stats">
    <span><strong style="color: #76B900;">3&times;</strong> Speedup</span>
    <span><strong style="color: #76B900;">FP8</strong> GEMM</span>
    <span><strong style="color: #76B900;">A100</strong> Optimized</span>
  </div>
</div>

<div class="cta-row">
  <a href="/en/getting-started/" class="cta-btn primary">Get Started</a>
  <a href="https://github.com/LessUp/triton-fused-ops" class="cta-btn ghost">View on GitHub</a>
</div>

## Features

<div class="feature-map">
  <div class="feature-card">
    <div class="feature-card-title">&#9889; Kernel Fusion</div>
    <div class="feature-card-desc">
      RMSNorm + RoPE fused in one kernel launch. Eliminates intermediate HBM round-trips.
    </div>
    <div class="feature-tags">
      <a href="/en/api/kernels" class="feature-tag">Core Kernels</a>
      <a href="/en/internals/kernel-design" class="feature-tag">Design</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">&#128290; FP8 Quantization</div>
    <div class="feature-card-desc">
      E4M3/E5M2-compatible FP8 GEMM pipeline with explicit scale management and overflow handling.
    </div>
    <div class="feature-tags">
      <a href="/en/api/quantization" class="feature-tag">Quantization</a>
      <a href="/en/guides/fp8-best-practices" class="feature-tag">Best Practices</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">&#127911; Auto-Tuning</div>
    <div class="feature-card-desc">
      TritonAutoTuner with configurable search space and persistent ConfigCache for optimal launch params.
    </div>
    <div class="feature-tags">
      <a href="/en/api/autotuner" class="feature-tag">Auto-Tuning</a>
      <a href="/en/api/models" class="feature-tag">Models</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">&#128202; Benchmarking</div>
    <div class="feature-card-desc">
      Built-in BenchmarkSuite with correctness verification and structured performance reports.
    </div>
    <div class="feature-tags">
      <a href="/en/api/benchmark" class="feature-tag">Benchmark</a>
      <a href="/en/guides/performance" class="feature-tag">Performance</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">&#129514; CPU-Testable References</div>
    <div class="feature-card-desc">
      Pure NumPy reference implementations for correctness validation without GPU hardware.
    </div>
    <div class="feature-tags">
      <a href="/en/internals/architecture" class="feature-tag">Architecture</a>
      <a href="/en/api/validation" class="feature-tag">Validation</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">&#127959; Modular Architecture</div>
    <div class="feature-card-desc">
      Strict separation of validation, compute reference, kernel, and benchmark layers.
    </div>
    <div class="feature-tags">
      <a href="/en/internals/architecture" class="feature-tag">Architecture</a>
      <a href="/en/api/validation" class="feature-tag">Contracts</a>
    </div>
  </div>
</div>

## Quick Start

<div class="quick-start">
  <div class="quick-start-title">Install</div>
  <div class="command-block">
    <code>pip install triton-fused-ops</code>
  </div>

  <div class="quick-start-title" style="margin-top: 16px;">Run your first kernel</div>

```python
import torch
from triton_ops import fused_rmsnorm_rope

x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
weight = torch.ones(4096, device="cuda", dtype=torch.float16)
cos = torch.randn(128, 64, device="cuda", dtype=torch.float16)
sin = torch.randn(128, 64, device="cuda", dtype=torch.float16)

y = fused_rmsnorm_rope(x, weight, cos, sin)
print(y.shape)  # torch.Size([2, 128, 4096])
```
</div>

## Performance Preview

<table class="perf-table">
  <thead>
    <tr>
      <th>Kernel</th>
      <th>Speedup vs PyTorch</th>
      <th>Memory Traffic Reduction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>fused_rmsnorm_rope</code></td>
      <td class="highlight">~3.0&times;</td>
      <td>~40%</td>
    </tr>
    <tr>
      <td><code>fused_gated_mlp</code></td>
      <td class="highlight">~1.5&times;</td>
      <td>~25%</td>
    </tr>
    <tr>
      <td><code>fp8_gemm</code></td>
      <td class="highlight">~1.3&times;</td>
      <td>~50% (weights)</td>
    </tr>
  </tbody>
</table>

<p style="font-size: 12px; color: var(--vp-c-text-3); margin-top: 8px;">
Measured on NVIDIA A100 SXM4 80GB, CUDA 12.1, PyTorch 2.1, Triton 2.1.
See <a href="/en/guides/performance">Performance Tuning</a> for methodology.
<a href="/en/guides/benchmark-visualization">View detailed charts &rarr;</a>
</p>

---

<p style="font-size: 13px; color: var(--vp-c-text-3); text-align: center; margin-top: 40px;">
Built on <a href="https://github.com/triton-lang/triton">Triton</a>,
<a href="https://github.com/pytorch/pytorch">PyTorch</a>, and
<a href="https://developer.nvidia.com/cuda-toolkit">CUDA</a>.
See <a href="/en/references/">References</a> for papers and related projects.
</p>
