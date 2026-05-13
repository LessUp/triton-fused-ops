---
title: Quick Start
description: "First working examples for the fused kernels and module wrappers"
---

# Quick Start

This page shows the shortest working path through the current public API.

### Quick-start flow

```mermaid
flowchart LR
    INSTALL["Install"] --> IMPORT["Import triton_ops"]
    IMPORT --> CHECK["Validate GPU"]
    CHECK --> RUN["Run Kernel"]
    RUN --> VERIFY["Verify Output"]

    style INSTALL fill:#21262d,stroke:#8b949e,color:#c9d1d9
    style IMPORT fill:#143,stroke:#76B900,color:#fff
    style CHECK fill:#1a1a2e,stroke:#ffc517,color:#ffc517
    style RUN fill:#143,stroke:#76B900,color:#fff
    style VERIFY fill:#0d2600,stroke:#76B900,color:#76B900
```

> **Figure 7.** Quick-start flow. Installation and import work on CPU-only environments. GPU validation is the first hardware-dependent step before running any kernel.

## Root imports

```python
import torch
from triton_ops import (
    fused_rmsnorm_rope,
    fused_gated_mlp,
    fp8_gemm,
    quantize_fp8,
    FusedRMSNormRoPE,
    FusedGatedMLP,
    FP8Linear,
)
```

## `fused_rmsnorm_rope`

```python
import torch
from triton_ops import fused_rmsnorm_rope

batch, seq_len, hidden_dim, head_dim = 2, 128, 4096, 64
x = torch.randn(batch, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float16)
cos = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
sin = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)

y = fused_rmsnorm_rope(x, weight, cos, sin)
print(y.shape)  # torch.Size([2, 128, 4096])
```

What matters:

- `x` must be 3D and contiguous.
- `weight.shape` must equal `(hidden_dim,)`.
- `cos` and `sin` must have the same shape.
- If `num_heads` is omitted, the code infers it from `hidden_dim / head_dim`.

## `fused_gated_mlp`

```python
import torch
from triton_ops import fused_gated_mlp

hidden_dim = 4096
intermediate_dim = 11008

x = torch.randn(2, 128, hidden_dim, device="cuda", dtype=torch.float16)
gate_weight = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16)
up_weight = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16)

y = fused_gated_mlp(x, gate_weight, up_weight, activation="silu")
print(y.shape)  # torch.Size([2, 128, 11008])
```

The current kernel implements:

```text
output = activation(gate_proj(x)) * up_proj(x)
```

Supported activations are `"silu"` and `"gelu"`.

## `fp8_gemm`

```python
import torch
from triton_ops import fp8_gemm, quantize_fp8

a = torch.randn(1024, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 2048, device="cuda", dtype=torch.float16)

# Option 1: let the library quantize inputs
c_auto = fp8_gemm(a, b)

# Option 2: quantize explicitly and pass scales
a_fp8, a_scale = quantize_fp8(a)
b_fp8, b_scale = quantize_fp8(b)
c_manual = fp8_gemm(a_fp8, b_fp8, a_scale, b_scale)

print(c_auto.shape, c_manual.shape)
```

Use `torch.float16` or `torch.bfloat16` outputs in practice. Those are the current half-precision output paths implemented by the Triton kernel.

## Module wrappers

```python
import torch
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP, FP8Linear

class DecoderBlock(torch.nn.Module):
    def __init__(self, hidden_dim=4096, num_heads=32, intermediate_dim=11008):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.norm = FusedRMSNormRoPE(hidden_dim, head_dim)
        self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim, activation="silu")
        self.proj = FP8Linear(hidden_dim, hidden_dim)

    def forward(self, x, cos, sin):
        normed = self.norm(x, cos, sin)
        mixed = self.proj(normed)
        mlp_out = self.mlp(normed)
        return mixed, mlp_out
```

`FP8Linear` lazily quantizes weights on the first forward pass and caches a transposed FP8 copy for later calls.

## Next

- [Examples](/en/getting-started/examples)
- [Integration Guide](/en/guides/integration)
- [Core Kernels API](/en/api/kernels)
