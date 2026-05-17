---
title: 性能可视化
description: "Triton Fused Ops 算子的可视化性能对比"
---

<script setup>
import BenchmarkVisualizationFigures from '@theme/components/BenchmarkVisualizationFigures.vue'
</script>

# 性能可视化

以下图表展示了仓库算子的代表性性能趋势。数据来自 NVIDIA A100 SXM4 80GB 环境。

<BenchmarkVisualizationFigures />

---

<p style="font-size: 12px; color: var(--vp-c-text-3);">
<strong>数据来源：</strong> NVIDIA A100 SXM4 80GB, CUDA 12.1, PyTorch 2.1, Triton 2.1。Latency 测量使用 <code>torch.cuda.synchronize()</code> 在计时区域前后同步，10 轮 warmup + 100 轮 benchmark。
</p>
