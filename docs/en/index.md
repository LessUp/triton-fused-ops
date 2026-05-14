---
layout: home
hero:
  name: Triton Fused Ops
  text: High-Performance GPU Kernels for Transformer Inference
  tagline: Production-ready Triton implementations of RMSNorm+RoPE fusion, Gated MLP fusion, and FP8 GEMM with auto-tuning infrastructure.
  actions:
    - theme: brand
      text: Getting Started
      link: /en/getting-started/
    - theme: alt
      text: Architecture
      link: /en/internals/architecture
    - theme: alt
      text: GitHub
      link: https://github.com/LessUp/triton-fused-ops
features:
  - icon: ⚡
    title: Kernel Fusion
    details: RMSNorm + RoPE fused in a single kernel launch. Eliminates intermediate HBM round-trips, achieving ~3× speedup over naive PyTorch implementation.
    link: /en/api/kernels
  - icon: 📐
    title: FP8 GEMM
    details: E4M3/E5M2-compatible FP8 quantized GEMM pipeline with explicit scale management, overflow handling, and 50% memory reduction for weights.
    link: /en/api/quantization
  - icon: 🔧
    title: Auto-Tuning
    details: TritonAutoTuner with configurable search space and persistent ConfigCache. Automatically discovers optimal launch parameters per hardware.
    link: /en/api/autotuner
  - icon: 📊
    title: Benchmarking
    details: Built-in BenchmarkSuite with correctness verification, structured PerformanceReport, and visualization utilities.
    link: /en/api/benchmark
  - icon: 🧪
    title: CPU References
    details: Pure NumPy reference implementations in triton_ops.compute. Validate correctness without GPU hardware.
    link: /en/internals/architecture
  - icon: 🏗️
    title: Modular Design
    details: "Strict separation: validation → compute reference → kernel → benchmark. Clean contracts, testable layers."
    link: /en/internals/architecture
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
