---
layout: home
---

<div class="home-header">
  <div class="home-header-left">
    <div class="home-logo">TFO</div>
    <div>
      <span class="home-title">Triton Fused Ops</span>
      <span class="home-subtitle">面向 Transformer 推理的高性能 GPU 融合算子库</span>
    </div>
  </div>
  <div class="home-nav">
    <a href="/zh/getting-started/">开始使用</a>
    <a href="https://github.com/LessUp/triton-fused-ops">GitHub</a>
    <a href="/en/">English</a>
  </div>
</div>

<div class="home-intro-row">
  <div class="home-intro">
    面向 LLM 推理的融合 GPU 算子库。Memory-bound &rarr; Compute-bound。本库提供生产级 Triton 实现，包括 RMSNorm+RoPE 融合、Gated MLP 融合、FP8 GEMM，以及自动调优和基准测试基础设施。
  </div>
  <div class="home-stats">
    <span><strong style="color: #76B900;">3&times;</strong> 加速</span>
    <span><strong style="color: #76B900;">FP8</strong> GEMM</span>
    <span><strong style="color: #76B900;">A100</strong> 优化</span>
  </div>
</div>

<div class="cta-row">
  <a href="/zh/getting-started/" class="cta-btn primary">开始使用</a>
  <a href="https://github.com/LessUp/triton-fused-ops" class="cta-btn ghost">GitHub 仓库</a>
</div>

## 功能特性

<div class="feature-map">
  <div class="feature-card">
    <div class="feature-card-title">&#9889; 算子融合</div>
    <div class="feature-card-desc">
      单次 kernel 启动完成 RMSNorm + RoPE 融合，消除中间 HBM 往返。
    </div>
    <div class="feature-tags">
      <a href="/zh/api/kernels" class="feature-tag">核心算子</a>
      <a href="/zh/internals/kernel-design" class="feature-tag">设计思路</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">&#128290; FP8 量化</div>
    <div class="feature-card-desc">
      兼容 E4M3/E5M2 的 FP8 GEMM 管线，支持显式 scale 管理与溢出处理。
    </div>
    <div class="feature-tags">
      <a href="/zh/api/quantization" class="feature-tag">量化</a>
      <a href="/zh/guides/fp8-best-practices" class="feature-tag">最佳实践</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">&#127911; 自动调优</div>
    <div class="feature-card-desc">
      TritonAutoTuner 支持可配置搜索空间与持久化 ConfigCache，自动寻找最优启动参数。
    </div>
    <div class="feature-tags">
      <a href="/zh/api/autotuner" class="feature-tag">自动调优</a>
      <a href="/zh/api/models" class="feature-tag">数据模型</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">&#128202; 基准测试</div>
    <div class="feature-card-desc">
      内置 BenchmarkSuite，一键完成正确性验证与结构化性能报告。
    </div>
    <div class="feature-tags">
      <a href="/zh/api/benchmark" class="feature-tag">基准测试</a>
      <a href="/zh/guides/performance" class="feature-tag">性能</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">&#129514; CPU 可测参考</div>
    <div class="feature-card-desc">
      纯 NumPy 参考实现，无需 GPU 硬件即可完成正确性验证。
    </div>
    <div class="feature-tags">
      <a href="/zh/internals/architecture" class="feature-tag">架构</a>
      <a href="/zh/api/validation" class="feature-tag">校验</a>
    </div>
  </div>

  <div class="feature-card">
    <div class="feature-card-title">&#127959; 模块化架构</div>
    <div class="feature-card-desc">
      验证层、计算参考层、内核层、基准层严格分离，边界清晰。
    </div>
    <div class="feature-tags">
      <a href="/zh/internals/architecture" class="feature-tag">架构</a>
      <a href="/zh/api/validation" class="feature-tag">契约</a>
    </div>
  </div>
</div>

## 快速开始

<div class="quick-start">
  <div class="quick-start-title">安装</div>
  <div class="command-block">
    <code>pip install triton-fused-ops</code>
  </div>

  <div class="quick-start-title" style="margin-top: 16px;">运行第一个 kernel</div>

```python
import torch
from triton_ops import fused_rmsnorm_rope

# 准备输入（需 CUDA）
x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
weight = torch.ones(4096, device="cuda", dtype=torch.float16)
cos = torch.randn(128, 64, device="cuda", dtype=torch.float16)
sin = torch.randn(128, 64, device="cuda", dtype=torch.float16)

# 融合 RMSNorm + RoPE
y = fused_rmsnorm_rope(x, weight, cos, sin)
print(y.shape)  # torch.Size([2, 128, 4096])
```
</div>

## 性能概览

<table class="perf-table">
  <thead>
    <tr>
      <th>算子</th>
      <th>相比 PyTorch 加速</th>
      <th>内存流量削减</th>
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
测试环境：NVIDIA A100 SXM4 80GB, CUDA 12.1, PyTorch 2.1, Triton 2.1。
测量方法详见 <a href="/zh/guides/performance">性能优化</a>。
<a href="/zh/guides/benchmark-visualization">查看详细图表 &rarr;</a>
</p>

---

<p style="font-size: 13px; color: var(--vp-c-text-3); text-align: center; margin-top: 40px;">
基于 <a href="https://github.com/triton-lang/triton">Triton</a>、
<a href="https://github.com/pytorch/pytorch">PyTorch</a> 和
<a href="https://developer.nvidia.com/cuda-toolkit">CUDA</a> 构建。
详见 <a href="/zh/references/">参考文献</a> 了解相关论文与开源项目。
</p>
