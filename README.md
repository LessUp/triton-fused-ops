# Triton Fused Operators Library

[![CI](https://github.com/LessUp/triton-fused-ops/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/triton-fused-ops/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/triton-fused-ops/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/triton-fused-ops/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Triton](https://img.shields.io/badge/Triton-2.1+-76B900?logo=nvidia&logoColor=white)

English | [简体中文](README.zh-CN.md) | [Project Page](https://lessup.github.io/triton-fused-ops/)

High-performance Triton operator library optimized for Transformer models, featuring operator fusion and FP8 quantization.

## Features

- **RMSNorm + RoPE Fusion** — Fuses normalization and rotary position encoding into a single kernel, reducing 3 HBM accesses to 1
- **Gated MLP Fusion** — Fuses gate projection and activation (SiLU/GELU)
- **FP8 Quantized GEMM** — 8-bit floating-point matmul with dynamic scaling, <1% precision loss
- **Auto-Tuning Framework** — Automatic search for optimal BLOCK_SIZE, num_warps, etc.
- **Benchmark Suite** — Correctness verification and performance comparison vs PyTorch/cuBLAS

## Installation

```bash
pip install -e ".[dev]"
```

Requirements: Python >= 3.9, PyTorch >= 2.0, Triton >= 2.1, CUDA >= 11.8

## Quick Start

### Functional API

```python
from triton_ops import fused_rmsnorm_rope, fused_gated_mlp, fp8_gemm

# RMSNorm + RoPE fusion
output = fused_rmsnorm_rope(x, weight, cos, sin)

# Gated MLP fusion
output = fused_gated_mlp(x, gate_weight, up_weight, activation='silu')

# FP8 GEMM (auto-quantize)
output = fp8_gemm(a, b)
```

### Module API

```python
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP, FP8Linear

class TransformerBlock(torch.nn.Module):
    def __init__(self, hidden_dim=4096, head_dim=64, intermediate_dim=11264):
        super().__init__()
        self.norm = FusedRMSNormRoPE(hidden_dim, head_dim)
        self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim, activation='silu')
        self.proj = FP8Linear(intermediate_dim, hidden_dim)
```

## Testing & Benchmarks

```bash
pytest tests/ -v                              # All tests
python -m tests.benchmarks.bench_rmsnorm_rope # RMSNorm+RoPE benchmark
python -m tests.benchmarks.bench_fp8_gemm     # FP8 GEMM benchmark
```

## Project Structure

```
triton_ops/
├── kernels/           # RMSNorm+RoPE, Gated MLP, FP8 GEMM, FP8 quantize
├── autotuner/         # Auto-tuning framework + config cache
├── benchmark/         # Benchmark suite + reports
├── api.py             # Convenience API
├── models.py          # Data models (TensorSpec, KernelMetrics)
├── validation.py      # Input validation
└── exceptions.py      # Custom exceptions
```

## Performance Highlights

- **RMSNorm + RoPE**: 90%+ memory bandwidth utilization via kernel fusion
- **FP8 GEMM**: 50% VRAM reduction vs FP16, <1% precision loss
- **Auto-Tuning**: Automatic optimal config search per hardware/problem size

## License

MIT License
