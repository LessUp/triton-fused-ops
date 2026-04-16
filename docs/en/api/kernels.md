---
layout: default
title: "Core Kernels API — Triton Fused Ops"
description: "API reference for core Triton fused kernels - RMSNorm+RoPE, Gated MLP, FP8 GEMM"
---

# Core Kernels API Reference

This document provides detailed API reference for the core Triton fused kernels.

---

## Table of Contents

- [fused_rmsnorm_rope](#fused_rmsnorm_rope)
- [fused_gated_mlp](#fused_gated_mlp)
- [fp8_gemm](#fp8_gemm)
- [FusedRMSNormRoPE](#fusedrmsnormrope)
- [FusedGatedMLP](#fusedgatedmlp)
- [FP8Linear](#fp8linear)

---

## fused_rmsnorm_rope

Apply fused RMSNorm + Rotary Position Embedding transformation.

Combines RMSNorm and RoPE into a single kernel launch, reducing memory bandwidth requirements by eliminating intermediate HBM writes.

### Syntax

```python
triton_ops.fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
    num_heads: Optional[int] = None,
) -> torch.Tensor
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Input tensor of shape `[batch, seq_len, hidden_dim]`. Must be CUDA tensor with dtype `float16` or `bfloat16`. |
| `weight` | `torch.Tensor` | RMSNorm weight of shape `[hidden_dim]`. Must match `x.dtype` and be on CUDA. |
| `cos` | `torch.Tensor` | Cosine position embeddings of shape `[seq_len, head_dim]`. Precomputed from position indices. |
| `sin` | `torch.Tensor` | Sine position embeddings of shape `[seq_len, head_dim]`. Precomputed from position indices. |
| `eps` | `float` | Small constant for numerical stability in RMSNorm. Default: `1e-6`. |
| `num_heads` | `Optional[int]` | Number of attention heads. If `None`, inferred from `hidden_dim / head_dim`. |

### Returns

`torch.Tensor` — Output tensor of shape `[batch, seq_len, hidden_dim]` with RMSNorm + RoPE applied.

### Raises

| Exception | Condition |
|-----------|-----------|
| `DeviceError` | CUDA is not available or tensor is not on CUDA. |
| `ShapeMismatchError` | Tensor shapes are incompatible. |
| `UnsupportedDtypeError` | Tensor dtype is not supported. |

### Mathematical Formula

```
RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
RoPE: y_rope = y * cos + rotate_half(y) * sin
```

### Example

```python
import torch
from triton_ops import fused_rmsnorm_rope

# Create inputs
batch, seq_len, hidden_dim, head_dim = 2, 128, 4096, 64
x = torch.randn(batch, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
weight = torch.ones(hidden_dim, device='cuda', dtype=torch.float16)

# Precompute position embeddings
positions = torch.arange(seq_len, device='cuda')
freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device='cuda') / head_dim))
angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
cos = torch.cos(angles).to(torch.float16)
sin = torch.sin(angles).to(torch.float16)

# Apply fused kernel
output = fused_rmsnorm_rope(x, weight, cos, sin)
```

---

## fused_gated_mlp

Apply fused Gated MLP transformation (SwiGLU/GeGLU).

Computes: `output = activation(gate_proj(x)) * up_proj(x)`

### Syntax

```python
triton_ops.fused_gated_mlp(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    activation: Literal["silu", "gelu"] = "silu",
) -> torch.Tensor
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `torch.Tensor` | Input tensor of shape `[batch, seq_len, hidden_dim]`. |
| `gate_weight` | `torch.Tensor` | Gate projection weight of shape `[intermediate_dim, hidden_dim]`. |
| `up_weight` | `torch.Tensor` | Up projection weight of shape `[intermediate_dim, hidden_dim]`. |
| `activation` | `Literal["silu", "gelu"]` | Activation function. `"silu"` for SwiGLU, `"gelu"` for GeGLU. Default: `"silu"`. |

### Returns

`torch.Tensor` — Output tensor of shape `[batch, seq_len, intermediate_dim]`.

### Example

```python
import torch
from triton_ops import fused_gated_mlp

# LLaMA-style configuration
hidden_dim = 4096
intermediate_dim = 11008  # ~2.67x hidden_dim

x = torch.randn(2, 128, hidden_dim, device='cuda', dtype=torch.float16)
gate_w = torch.randn(intermediate_dim, hidden_dim, device='cuda', dtype=torch.float16)
up_w = torch.randn(intermediate_dim, hidden_dim, device='cuda', dtype=torch.float16)

# SwiGLU
output = fused_gated_mlp(x, gate_w, up_w, activation='silu')
```

---

## fp8_gemm

Perform FP8 quantized matrix multiplication.

If inputs are not already in FP8 format, they will be quantized automatically.

### Syntax

```python
triton_ops.fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `torch.Tensor` | First matrix of shape `[M, K]`. Can be FP8 (`uint8`) or float (`float16`, `bfloat16`, `float32`). |
| `b` | `torch.Tensor` | Second matrix of shape `[K, N]`. Can be FP8 (`uint8`) or float. |
| `a_scale` | `Optional[torch.Tensor]` | Scale factor for A (required if A is FP8, computed if float). |
| `b_scale` | `Optional[torch.Tensor]` | Scale factor for B (required if B is FP8, computed if float). |
| `output_dtype` | `torch.dtype` | Output data type. Default: `torch.float16`. |

### Returns

`torch.Tensor` — Result matrix of shape `[M, N]` in `output_dtype`.

### Example

```python
import torch
from triton_ops import fp8_gemm

# Automatic quantization
a = torch.randn(1024, 4096, device='cuda', dtype=torch.float16)
b = torch.randn(4096, 2048, device='cuda', dtype=torch.float16)
c = fp8_gemm(a, b)  # Auto-quantizes both inputs

# Pre-quantized inputs
from triton_ops import quantize_fp8
a_fp8, a_scale = quantize_fp8(a)
b_fp8, b_scale = quantize_fp8(b)
c = fp8_gemm(a_fp8, b_fp8, a_scale, b_scale)
```

---

## FusedRMSNormRoPE

PyTorch `nn.Module` wrapper for fused RMSNorm + RoPE.

### Syntax

```python
triton_ops.FusedRMSNormRoPE(
    hidden_dim: int,
    head_dim: int,
    eps: float = 1e-6,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `hidden_dim` | `int` | Hidden dimension size. |
| `head_dim` | `int` | Head dimension for RoPE. |
| `eps` | `float` | Epsilon for RMSNorm. Default: `1e-6`. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `weight` | `nn.Parameter` | Learnable RMSNorm weight, shape `[hidden_dim]`. |

### Forward Method

```python
forward(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor
```

### Example

```python
import torch
from triton_ops import FusedRMSNormRoPE

# Create module
norm = FusedRMSNormRoPE(hidden_dim=4096, head_dim=64).cuda()

# Forward pass
x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
cos = torch.randn(128, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(128, 64, device='cuda', dtype=torch.float16)
output = norm(x, cos, sin)
```

---

## FusedGatedMLP

PyTorch `nn.Module` wrapper for fused Gated MLP.

### Syntax

```python
triton_ops.FusedGatedMLP(
    hidden_dim: int,
    intermediate_dim: int,
    activation: Literal["silu", "gelu"] = "silu",
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `hidden_dim` | `int` | Input hidden dimension. |
| `intermediate_dim` | `int` | FFN intermediate dimension. |
| `activation` | `Literal["silu", "gelu"]` | Activation function. Default: `"silu"`. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `gate_weight` | `nn.Parameter` | Gate projection weight, shape `[intermediate_dim, hidden_dim]`. |
| `up_weight` | `nn.Parameter` | Up projection weight, shape `[intermediate_dim, hidden_dim]`. |

### Example

```python
import torch
from triton_ops import FusedGatedMLP

# LLaMA-style MLP
mlp = FusedGatedMLP(
    hidden_dim=4096,
    intermediate_dim=11008,
    activation='silu'
).cuda().half()

x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
output = mlp(x)
```

---

## FP8Linear

Linear layer with FP8 quantized weights.

### Syntax

```python
triton_ops.FP8Linear(
    in_features: int,
    out_features: int,
    bias: bool = False,
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `in_features` | `int` | Input feature dimension. |
| `out_features` | `int` | Output feature dimension. |
| `bias` | `bool` | Whether to include bias. Default: `False`. |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `weight` | `nn.Parameter` | FP16 weight, quantized to FP8 on first forward. |
| `weight_fp8` | `Tensor` | FP8 quantized weight (lazy-initialized). |
| `weight_scale` | `Tensor` | Scale factor for FP8 weight. |

### Example

```python
import torch
from triton_ops import FP8Linear

# Create FP8 linear layer
linear = FP8Linear(in_features=4096, out_features=4096).cuda()

# Forward pass (weights quantized on first call)
x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
output = linear(x)
```

---

<div align="center">

**[⬆ Back to Top](#core-kernels-api-reference)** | **[← Back to API Index](./)**

</div>
