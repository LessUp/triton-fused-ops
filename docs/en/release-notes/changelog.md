# Changelog

This page documents the changes in each release of Triton Fused Ops.

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## 1.0.1 (2026-04-27)

- FP8 GEMM scale validation: now requires scale factor when inputs are already FP8
- Optional type annotations in `fp8_gemm.py`, `fp8_quantize.py`, and `api.py`
- Type syntax consistency: unified use of `Tuple` from typing module
- Exception handling in `tuner.py`: narrowed from `Exception` to `RuntimeError, OSError`
- TOCTOU race condition in `cache.py`: removed redundant `exists()` check

- Archived completed OpenSpec change `prepare-project-for-archive`
- Removed `_bmad/` and `_bmad-output/` residual directories
- Removed `_bmad` entries from `.gitignore`
- Configured Git hooks path to `.githooks`

- pytest-cov configuration with 70% coverage threshold

---

## [Unreleased]

---

## 1.0.0 (2026-04-16)

### 🎉 First Stable Release

We are excited to announce the first stable release of Triton Fused Ops!

### ✨ Highlights

- **Production-Ready**: All kernels have been thoroughly tested and optimized
- **Bilingual Documentation**: Complete English and Chinese documentation
- **Professional Changelog**: Adopted Keep a Changelog standard
- **Comprehensive Testing**: Full test suite with Hypothesis property-based testing

### 🚀 Core Features

#### Fused Kernels

| Kernel | Speedup | Memory Savings | Status |
|:-------|:-------:|:--------------:|:-------|
| **RMSNorm + RoPE Fusion** | ~3x | 50% fewer HBM writes | ✅ Stable |
| **Gated MLP Fusion** | ~1.5x | 1 intermediate tensor less | ✅ Stable |
| **FP8 GEMM** | ~1.4x | 50% weight storage | ✅ Stable |

#### Infrastructure

- **Auto-Tuning Framework**: Automatic kernel configuration optimization
  - Configurable search space
  - Persistent result caching
  - Multiple tuning strategies
  - Pre-defined configuration spaces for all kernels

- **Benchmark Suite**: Comprehensive performance measurement
  - Correctness verification against PyTorch reference
  - Performance measurement with synchronization
  - Report generation with metrics

- **Test Suite**: Comprehensive testing infrastructure
  - Unit tests for all kernels
  - Property-based tests using Hypothesis
  - Edge case coverage
  - CI/CD integration

### 📚 Documentation

- Complete API documentation with examples
- User guides for integration, performance tuning, and FP8 best practices
- Internal architecture and kernel design documentation
- Bilingual support (English/Chinese)

### ⚠️ Breaking Changes

None - this is the first stable release.

### 📦 Installation

```bash
pip install triton-fused-ops
```

### 🙏 Acknowledgments

Thanks to all contributors and the OpenAI Triton team for their amazing work.

---

## 0.2.0 (2026-03-09)

#### New Features
- **SwiGLU Correctness Fix**: Corrected the activation application order in gated MLP to follow the standard SwiGLU formula: `output = activation(gate_proj(x)) * up_proj(x)`
- **FP8Linear Weight Transpose Caching**: Pre-transpose and cache weights in `FP8Linear` to avoid `.t().contiguous()` on every forward pass
- **Improved Input Validation**: Added comprehensive validation with helpful error messages

#### Infrastructure
- GitHub Actions CI/CD pipeline with automated testing
- GitHub Pages documentation site
- Comprehensive test suite with Hypothesis property-based testing

- Refactored kernel launch patterns for better error handling
- Improved memory access patterns in fused kernels
- Updated minimum version requirements (PyTorch 2.0+, Triton 2.1+)

- **RMSNorm batch_idx computation**: Fixed incorrect batch index calculation in the fused RMSNorm + RoPE kernel
- **FP8 GEMM block sizes**: Replaced hardcoded values with adaptive heuristics
- Memory leaks in long-running autotuner sessions

---

## 0.1.0 (2024-01-01)

#### Fused Kernels
- **RMSNorm + RoPE Fusion**: Fused kernel combining RMS normalization with Rotary Position Embedding
  - Supports both functional and module APIs
  - Configurable epsilon for numerical stability
  - Optimized memory access patterns
  - Achieves ~3x speedup over separate operations

- **Gated MLP Fusion**: Fused kernel for gated MLP layers
  - Supports SiLU (SwiGLU) and GELU (GeGLU) activation functions
  - Single-pass computation for gate and up projections
  - Reduced memory bandwidth requirements
  - ~1.5x speedup over standard implementation

- **FP8 GEMM**: FP8 quantized matrix multiplication
  - E4M3 format support
  - Automatic scale computation
  - FP32 accumulation for numerical stability
  - ~1.4x speedup with 50% memory savings

- **FP8 Quantization**: Utilities for FP8 quantization
  - Dynamic range computation
  - Scale factor calculation
  - Overflow handling with automatic retry
  - Dequantization support

#### Infrastructure
- **Autotuner Framework**: Automatic kernel configuration optimization
- **Benchmark Suite**: Comprehensive performance measurement
- **Test Suite**: Comprehensive testing infrastructure

#### Documentation
- README with installation and usage instructions
- API documentation with examples
- Contributing guidelines
- Code of Conduct
- Chinese documentation (README.zh-CN.md)

### Technical Details

| Requirement | Version |
|:------------|:--------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.0 |
| Triton | ≥ 2.1 |
| CUDA | ≥ 11.8 |
| GPU Architecture | Ampere (SM80+) |

---

## Version History Summary

| Version | Date | Highlights |
|:--------|:-----|:-----------|
| 1.0.0 | 2026-04-16 | First stable release, bilingual docs |
| 0.2.0 | 2026-03-09 | SwiGLU fix, FP8Linear optimization, CI/CD |
| 0.1.0 | 2024-01-01 | Initial release with all core kernels |

---

[Unreleased]: https://github.com/LessUp/triton-fused-ops/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/LessUp/triton-fused-ops/releases/tag/v1.0.0
[0.2.0]: https://github.com/LessUp/triton-fused-ops/releases/tag/v0.2.0
[0.1.0]: https://github.com/LessUp/triton-fused-ops/releases/tag/v0.1.0
