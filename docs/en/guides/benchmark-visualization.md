---
title: Benchmark Visualization
description: "Visual performance comparisons of Triton Fused Ops kernels"
---

<script setup>
import BenchmarkVisualizationFigures from '@theme/components/BenchmarkVisualizationFigures.vue'
</script>

# Benchmark Visualization

These charts illustrate representative performance trends measured on the repository kernels. The numbers are directional references from an NVIDIA A100 SXM4 80GB environment.

<BenchmarkVisualizationFigures />

---

<p style="font-size: 12px; color: var(--vp-c-text-3);">
<strong>Data source:</strong> Measured on NVIDIA A100 SXM4 80GB, CUDA 12.1, PyTorch 2.1, Triton 2.1. Latency measured with <code>torch.cuda.synchronize()</code> before and after the timed region, 10 warmup runs + 100 benchmark runs.
</p>
