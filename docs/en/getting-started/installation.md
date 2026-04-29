---
layout: default
title: Installation
parent: Getting Started
grand_parent: Documentation
nav_order: 1
description: "Environment requirements and installation workflow for Triton Fused Ops"
---

# Installation

Use this page to prepare a working environment and run the first validation steps.

## Requirements

| Area | Baseline | Notes |
|:--|:--|:--|
| Python | `>=3.9` | The package metadata targets Python 3.9+ |
| PyTorch | `>=2.0.0` | CUDA build required for actual Triton kernel execution |
| Triton | `>=2.1.0` | OpenAI Triton |
| GPU | CUDA-capable NVIDIA GPU | Needed for kernels and GPU benchmarks |

## Install from source

```bash
git clone https://github.com/LessUp/triton-fused-ops.git
cd triton-fused-ops
pip install -e ".[dev]"
```

If you only need the package itself:

```bash
pip install -e .
```

If you use `uv`:

```bash
uv pip install -e ".[dev]"
```

## CPU-safe baseline checks

These checks do not require running Triton kernels and are suitable for CI or CPU-only validation paths:

```bash
python -c "import triton_ops; print(triton_ops.__version__)"
ruff format --check .
ruff check .
mypy triton_ops/
pytest tests/ -v -k "not cuda and not gpu" --ignore=tests/benchmarks/
python3 -m build
```

## GPU smoke test

```python
import torch
from triton_ops import fused_rmsnorm_rope

assert torch.cuda.is_available()

batch, seq_len, hidden_dim, head_dim = 2, 128, 4096, 64
x = torch.randn(batch, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float16)
cos = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
sin = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)

y = fused_rmsnorm_rope(x, weight, cos, sin)
print(y.shape, y.dtype)
```

Notes:

- The current implementation accepts `cos` and `sin` in shape `[seq_len, head_dim]`.
- Validation also accepts cached 4D RoPE tensors in shape `[1, seq_len, 1, head_dim]`.
- Runtime validation expects CUDA tensors, supported floating dtypes, and contiguous inputs.

## Environment sanity check

```python
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name())
```

## Common problems

### `CUDA is not available`

Typical causes:

- PyTorch was installed without CUDA support.
- The active Python environment cannot see the expected NVIDIA driver/runtime.

Typical recovery:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### `DeviceError` on kernel calls

The exported kernels check that tensors are on CUDA before launching. Move all inputs to the same CUDA device and keep them contiguous.

### `UnsupportedDtypeError` or shape validation failures

Use the API pages for the exact constraints:

- `fused_rmsnorm_rope`: 3D `x`, 1D `weight`, 2D or 4D RoPE cache.
- `fused_gated_mlp`: 3D `x`, 2D weights, activation in `{"silu", "gelu"}`.
- `fp8_gemm`: 2D matrices; pre-quantized `uint8` inputs require matching scale tensors.

## Next

- [Quick Start]({{ '/docs/en/getting-started/quickstart/' | relative_url }})
- [Examples]({{ '/docs/en/getting-started/examples/' | relative_url }})
- [Core Kernels API]({{ '/docs/en/api/kernels/' | relative_url }})
