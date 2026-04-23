# Triton Fused Ops

<div align="center">

[![CI](https://github.com/LessUp/triton-fused-ops/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/triton-fused-ops/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/triton-fused-ops/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/triton-fused-ops/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Triton 2.1+](https://img.shields.io/badge/Triton-2.1+-76B900?logo=nvidia&logoColor=white)

High-performance Triton kernels for Transformer inference workloads.

[📖 Docs](https://lessup.github.io/triton-fused-ops/) | [🇨🇳 中文](README.zh-CN.md) | [💡 Examples](examples/) | [🤝 Contributing](CONTRIBUTING.md)

</div>

---

## What this repository provides

`triton-fused-ops` focuses on three kernel families:

- `fused_rmsnorm_rope`: RMSNorm + RoPE in one kernel
- `fused_gated_mlp`: gated MLP fusion (SiLU/GELU)
- `fp8_gemm` + quantization helpers: FP8 matrix multiplication pipeline

The goal is to reduce redundant memory traffic and keep a practical, testable integration surface.

## Runtime boundaries

- **GPU is required** for Triton kernel execution and performance benchmarking.
- **CPU-only environments** can still run import/type/lint/unit checks (CI uses this path).
- Performance numbers depend on GPU architecture, model shape, and batch size.

## Installation

```bash
git clone https://github.com/LessUp/triton-fused-ops.git
cd triton-fused-ops
pip install -e ".[dev]"
```

## Quick checks

Import check (works in CPU-only environments):

```bash
python -c "import triton_ops; print(triton_ops.__version__)"
```

CPU-safe validation baseline:

```bash
ruff format --check .
ruff check .
mypy triton_ops/
pytest tests/ -v -k "not cuda and not gpu" --ignore=tests/benchmarks/
python3 -m build
```

## Minimal usage (GPU)

```python
import torch
from triton_ops import fused_rmsnorm_rope

x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
weight = torch.ones(4096, device="cuda", dtype=torch.float16)
cos = torch.randn(128, 128, device="cuda", dtype=torch.float16)
sin = torch.randn(128, 128, device="cuda", dtype=torch.float16)

y = fused_rmsnorm_rope(x, weight, cos, sin)
print(y.shape)
```

## Benchmark note

Representative benchmark snapshots in this repository are measured on NVIDIA A100 (CUDA 12.1) and are intended as directional references, not universal guarantees.

| Kernel | Typical speedup vs unfused/reference path |
|:--|:--:|
| `fused_rmsnorm_rope` | up to ~3x |
| `fused_gated_mlp` | ~1.3x–1.8x |
| `fp8_gemm` | ~1.2x–1.5x |

See `tests/benchmarks/` and docs for setup details.

## Development workflow

This repository is **OpenSpec-driven** for non-trivial work:

1. Create/select an OpenSpec change
2. Complete proposal/design/specs/tasks
3. Implement task-by-task
4. Run review and validation

See:

- [`AGENTS.md`](AGENTS.md)
- [`CLAUDE.md`](CLAUDE.md)
- [`openspec/README.md`](openspec/README.md)

## License

MIT.
