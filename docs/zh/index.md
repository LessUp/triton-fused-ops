---
layout: home
hero:
  name: Triton Fused Ops
  text: 面向 Transformer 推理的高性能 GPU 融合算子库
  tagline: 生产级 Triton 实现：RMSNorm+RoPE 融合、Gated MLP 融合、FP8 GEMM，配备自动调优基础设施。
  actions:
    - theme: brand
      text: 开始使用
      link: /zh/getting-started/
    - theme: alt
      text: 架构设计
      link: /zh/internals/architecture
    - theme: alt
      text: GitHub
      link: https://github.com/LessUp/triton-fused-ops
features:
  - icon: ⚡
    title: 算子融合
    details: 单次 kernel 启动完成 RMSNorm + RoPE 融合，消除中间 HBM 往返，相比朴素 PyTorch 实现加速约 3 倍。
    link: /zh/api/kernels
  - icon: 📐
    title: FP8 GEMM
    details: 兼容 E4M3/E5M2 的 FP8 量化 GEMM 管线，显式 scale 管理、溢出处理，权重内存削减 50%。
    link: /zh/api/quantization
  - icon: 🔧
    title: 自动调优
    details: TritonAutoTuner 支持可配置搜索空间与持久化 ConfigCache，自动发现每类硬件的最优启动参数。
    link: /zh/api/autotuner
  - icon: 📊
    title: 基准测试
    details: 内置 BenchmarkSuite，一键完成正确性验证、结构化 PerformanceReport 与可视化。
    link: /zh/api/benchmark
  - icon: 🧪
    title: CPU 参考实现
    details: triton_ops.compute 中的纯 NumPy 参考实现，无需 GPU 硬件即可验证正确性。
    link: /zh/internals/architecture
  - icon: 🏗️
    title: 模块化设计
    details: "严格分层：validation → compute reference → kernel → benchmark。边界清晰，层层可测。"
    link: /zh/internals/architecture
---

<style>
/* Homepage-specific enhancements */
.VPHero .name {
  background: linear-gradient(120deg, #76B900 0%, #5a8a00 100%);
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
}
.VPHero .text {
  font-size: 18px !important;
  font-weight: 500 !important;
}
.VPFeature {
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
.VPFeature:hover {
  border-color: var(--vp-c-brand-1) !important;
  box-shadow: 0 0 16px rgba(118, 185, 0, 0.15);
}
</style>
