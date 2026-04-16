---
layout: default
title: "Architecture — Triton Fused Ops"
description: "Triton Fused Ops library architecture overview"
---

# Architecture Overview

Understanding the overall architecture of Triton Fused Ops.

---

## High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Kernels    │  │  Autotuner   │  │  Benchmark   │          │
│  │   (api.py)   │  │   (tuner)    │  │   (suite)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Triton Kernel Layer                           │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │
│  │ rmsnorm_rope   │  │  gated_mlp     │  │   fp8_gemm     │     │
│  │   (triton)     │  │   (triton)     │  │   (triton)     │     │
│  └────────────────┘  └────────────────┘  └────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GPU Hardware                               │
│              CUDA / PTX / SASS Instructions                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Organization

```
triton_ops/
├── __init__.py          # Public API exports
├── api.py               # Clean functional API
├── models.py            # Data models and types
├── exceptions.py        # Custom exceptions
├── utils.py             # Utility functions
├── kernels/             # Triton kernel implementations
│   ├── rmsnorm_rope.py
│   ├── gated_mlp.py
│   ├── fp8_gemm.py
│   └── fp8_quantize.py
├── autotuner/           # Auto-tuning framework
│   ├── tuner.py
│   ├── configs.py
│   └── cache.py
└── benchmark/           # Benchmark suite
    ├── suite.py
    ├── correctness.py
    └── report.py
```

---

## Design Principles

### 1. Separation of Concerns

| Layer | Responsibility |
|:------|:---------------|
| **API Layer** | User-facing interface, input validation |
| **Kernel Layer** | Low-level Triton implementations |
| **Tuning Layer** | Configuration optimization |
| **Hardware** | Actual computation |

### 2. Lazy Loading

Kernels and configurations are loaded on first use:

```python
# Kernels are not compiled until first call
from triton_ops import fused_rmsnorm_rope

# First call triggers Triton JIT compilation
output = fused_rmsnorm_rope(x, weight, cos, sin)

# Subsequent calls use cached binary
```

### 3. Type Safety

Comprehensive type hints throughout:

```python
def fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    ...
```

---

## Kernel Registration

```python
# triton_ops/__init__.py

# Kernels
from .kernels.rmsnorm_rope import (
    fused_rmsnorm_rope,
    FusedRMSNormRoPE,
)
from .kernels.gated_mlp import (
    fused_gated_mlp,
    FusedGatedMLP,
)
from .kernels.fp8_gemm import (
    fp8_gemm,
    FP8Linear,
)
from .kernels.fp8_quantize import (
    quantize_fp8,
    dequantize_fp8,
    quantize_fp8_with_overflow_handling,
)

# Autotuner
from .autotuner.tuner import (
    TritonAutoTuner,
    ConfigCache,
)
from .autotuner.configs import (
    RMSNORM_ROPE_CONFIGS,
    GATED_MLP_CONFIGS,
    FP8_GEMM_CONFIGS,
)

__all__ = [
    # Kernels
    "fused_rmsnorm_rope",
    "FusedRMSNormRoPE",
    "fused_gated_mlp",
    "FusedGatedMLP",
    "fp8_gemm",
    "FP8Linear",
    # Quantization
    "quantize_fp8",
    "dequantize_fp8",
    "quantize_fp8_with_overflow_handling",
    # Autotuner
    "TritonAutoTuner",
    "ConfigCache",
    "RMSNORM_ROPE_CONFIGS",
    "GATED_MLP_CONFIGS",
    "FP8_GEMM_CONFIGS",
]
```

---

## Error Handling

### Exception Hierarchy

```
TritonOpsError (base)
├── DeviceError
│   └── CUDA unavailable
├── ShapeMismatchError
│   └── Tensor shape incompatibility
├── DtypeError
│   └── Unsupported data type
├── TuningError
│   └── Auto-tuning failure
└── NumericalError
    └── Overflow in quantization
```

### Usage Example

```python
from triton_ops import fused_rmsnorm_rope, DeviceError, ShapeMismatchError

try:
    output = fused_rmsnorm_rope(x, weight, cos, sin)
except DeviceError as e:
    print(f"CUDA error: {e}")
except ShapeMismatchError as e:
    print(f"Shape error: {e}")
```

---

## Extension Points

### Adding a New Kernel

1. Implement Triton kernel in `kernels/`
2. Add functional API in `kernels/<name>.py`
3. Add module wrapper (optional)
4. Export in `__init__.py`
5. Add configuration space in `autotuner/configs.py`
6. Add tests in `tests/`

### Adding Autotuner Support

```python
# In autotuner/configs.py

MY_KERNEL_CONFIGS = {
    "BLOCK_M": [64, 128, 256],
    "BLOCK_N": [64, 128, 256],
    "num_warps": [4, 8],
}

# In __init__.py
from .autotuner.configs import MY_KERNEL_CONFIGS
__all__.append("MY_KERNEL_CONFIGS")
```

---

<div align="center">

**[⬆ Back to Top](#architecture-overview)** | **[← Back to Internals](../)**

</div>
