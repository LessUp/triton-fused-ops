# Triton Fused Ops

<div align="center">

[![CI](https://github.com/LessUp/triton-fused-ops/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/triton-fused-ops/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/triton-fused-ops/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/triton-fused-ops/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Triton 2.1+](https://img.shields.io/badge/Triton-2.1+-76B900?logo=nvidia&logoColor=white)
![CUDA 12.1+](https://img.shields.io/badge/CUDA-12.1+-76B900?logo=nvidia&logoColor=white)
![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-261230?logo=ruff&logoColor=white)
![Types: mypy](https://img.shields.io/badge/Types-mypy-2A6DB5?logo=python&logoColor=white)

**Fused GPU kernels for LLM inference. Memory-bound &rarr; Compute-bound.**

[📖 Docs](https://lessup.github.io/triton-fused-ops/) | [🇨🇳 中文](README.zh-CN.md) | [💡 Examples](examples/) | [🤝 Contributing](CONTRIBUTING.md)

</div>

---

## Why this repo stands out

- **Operator fusion with correctness guarantees** — every kernel ships with CPU-testable NumPy reference implementations, not just speed claims.
- **Production-ready FP8 GEMM pipeline** — explicit scale management and overflow handling, not toy quantization examples.
- **Latency-driven autotuner with persistent config cache** — `TritonAutoTuner` + `ConfigCache`, not one-off benchmark scripts.
- **OpenSpec-driven development** — every non-trivial change is design-documented before code, not YOLO-coded.

## Architecture

```
User API (triton_ops.__init__)
    ├── Validation Layer (device, dtype, shape, contiguity)
    ├── Compute Reference Layer (NumPy, CPU-testable)
    ├── Kernel Layer (Triton, GPU)
    └── Tooling Layer (autotuner, benchmark, performance metrics)
```

See [Architecture](https://lessup.github.io/triton-fused-ops/en/internals/architecture) and [Kernel Design](https://lessup.github.io/triton-fused-ops/en/internals/kernel-design) docs for details.

## Quick Start

```bash
git clone https://github.com/LessUp/triton-fused-ops.git
cd triton-fused-ops
pip install -e ".[dev]"
```

**CPU-only validation** (no GPU required):

```bash
ruff format --check . && ruff check . && mypy triton_ops/
pytest tests/ -v -k "not cuda and not gpu" --ignore=tests/benchmarks/
python3 -m build
```

**Full GPU benchmark** (requires CUDA):

```python
import torch
from triton_ops import fused_rmsnorm_rope, BenchmarkSuite
from triton_ops.kernels.rmsnorm_rope import fused_rmsnorm_rope_reference

x = torch.randn(2, 2048, 4096, device="cuda", dtype=torch.float16)
suite = BenchmarkSuite(warmup_runs=10, benchmark_runs=100)
result = suite.benchmark_kernel(
    fused_rmsnorm_rope, fused_rmsnorm_rope_reference,
    "fused_rmsnorm_rope", (2, 2048, 4096), x, ...
)
print(result.metrics.latency_ms)
```

## Performance

Representative numbers on NVIDIA A100 SXM4 80GB (CUDA 12.1, PyTorch 2.1, Triton 2.1). Methodology: 10 warmup runs + 100 benchmark runs with `torch.cuda.synchronize()` before and after timing.

| Kernel | Speedup vs PyTorch | Memory Traffic Reduction |
|:--|:--:|:--:|
| `fused_rmsnorm_rope` | up to ~3.0&times; | ~40% |
| `fused_gated_mlp` | ~1.3x&ndash;1.8&times; | ~25% |
| `fp8_gemm` | ~1.2x&ndash;1.5&times; | ~50% (weights) |

See [Benchmark Visualization](https://lessup.github.io/triton-fused-ops/en/guides/benchmark-visualization) for interactive charts.

## Documentation Index

| Section | Best For | Key Takeaway |
|:--|:--|:--|
| [Getting Started](https://lessup.github.io/triton-fused-ops/en/getting-started/) | First-time users | 5-minute first kernel run |
| [Kernel Design](https://lessup.github.io/triton-fused-ops/en/internals/kernel-design) | Interview prep | Fusion patterns, tiling, memory optimization |
| [Performance](https://lessup.github.io/triton-fused-ops/en/guides/performance) | Tuning practitioners | Correct timing, bottleneck analysis |
| [References](https://lessup.github.io/triton-fused-ops/en/references/) | Deep learning researchers | Papers, projects, tech stack landscape |

## Development

This repository is **OpenSpec-driven** for non-trivial work. See [`AGENTS.md`](AGENTS.md), [`CLAUDE.md`](CLAUDE.md), and [`openspec/README.md`](openspec/README.md).

## Citation

```bibtex
@software{triton_fused_ops,
  title = {Triton Fused Ops: High-Performance GPU Kernels for Transformer Inference},
  author = {LessUp},
  year = {2025},
  url = {https://github.com/LessUp/triton-fused-ops},
  note = {Built on OpenAI Triton, PyTorch, and CUDA}
}
```

## Acknowledgements

- [OpenAI Triton](https://github.com/triton-lang/triton) for the compiler and Python DSL
- [PyTorch](https://github.com/pytorch/pytorch) for the tensor runtime
- [NVIDIA](https://developer.nvidia.com/) for CUDA, FP8 hardware, and performance tooling

## License

MIT.
