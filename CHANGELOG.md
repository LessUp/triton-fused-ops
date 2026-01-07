# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

### Changed
- Nothing yet

### Fixed
- Nothing yet

## [0.1.0] - 2024-01-01

### Added

#### Fused Kernels
- **RMSNorm + RoPE Fusion**: Fused kernel combining RMS normalization with Rotary Position Embedding
  - Supports both functional and module APIs
  - Configurable epsilon for numerical stability
  - Optimized memory access patterns

- **Gated MLP Fusion**: Fused kernel for gated MLP layers
  - Supports SiLU (SwiGLU) and GELU (GeGLU) activation functions
  - Single-pass computation for gate and up projections
  - Reduced memory bandwidth requirements

- **FP8 GEMM**: FP8 quantized matrix multiplication
  - Per-tensor and per-token scaling support
  - E4M3 and E5M2 format support
  - Automatic scale computation

- **FP8 Quantization**: Utilities for FP8 quantization
  - Dynamic range computation
  - Scale factor calculation
  - Dequantization support

#### Infrastructure
- Autotuner framework for kernel optimization
  - Configurable search space
  - Result caching
  - Multiple tuning strategies

- Benchmark suite
  - Correctness verification
  - Performance measurement
  - Report generation

- Comprehensive test suite
  - Unit tests for all kernels
  - Property-based tests
  - Edge case coverage

#### Documentation
- README with installation and usage instructions
- API documentation with examples
- Contributing guidelines
- Code of Conduct

### Technical Details
- Minimum Python version: 3.9
- Minimum PyTorch version: 2.0
- Minimum Triton version: 2.1
- Supported GPU architectures: Ampere (SM80+), Ada Lovelace, Hopper

[Unreleased]: https://github.com/username/triton-fused-ops/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/username/triton-fused-ops/releases/tag/v0.1.0
