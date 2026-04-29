---
layout: default
title: Core Kernels
parent: API Reference
grand_parent: Documentation
nav_order: 1
description: "Reference for the fused kernel entry points and module wrappers"
---

# Core Kernels

This page documents the compute-heavy entry points exported from `triton_ops`.

## `fused_rmsnorm_rope`

```python
fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
    num_heads: int | None = None,
) -> torch.Tensor
```

Purpose:

- Apply RMSNorm and RoPE in one kernel launch.
- Avoid materializing the normalized intermediate back to HBM.

Input contract:

- `x` must be a contiguous CUDA tensor with shape `[batch, seq_len, hidden_dim]`.
- `weight` must be a contiguous CUDA tensor with shape `[hidden_dim]`.
- `cos` and `sin` must be contiguous CUDA tensors with matching shapes.
- Supported RoPE cache shapes are:
  - `[seq_len, head_dim]`
  - `[1, seq_len, 1, head_dim]`
- `head_dim` must be even.
- If `num_heads` is omitted, the function infers it from `hidden_dim / head_dim`.

Output:

- Same shape as `x`.
- Same dtype as `x`.

Common failures:

- `DeviceError` when tensors are not on CUDA.
- `ShapeMismatchError` when shapes are inconsistent.
- `UnsupportedDtypeError` when the tensors are not floating dtypes accepted by validation.

Example:

```python
import torch
from triton_ops import fused_rmsnorm_rope

x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
weight = torch.ones(4096, device="cuda", dtype=torch.float16)
cos = torch.randn(128, 64, device="cuda", dtype=torch.float16)
sin = torch.randn(128, 64, device="cuda", dtype=torch.float16)

y = fused_rmsnorm_rope(x, weight, cos, sin)
```

## `FusedRMSNormRoPE`

```python
FusedRMSNormRoPE(hidden_dim: int, head_dim: int, eps: float = 1e-6)
```

This wrapper owns the RMSNorm weight parameter and still expects external `cos` and `sin` tensors at call time:

```python
module = FusedRMSNormRoPE(4096, 64).cuda()
out = module(x, cos, sin)
```

Integration note:

- This is not a drop-in replacement for a standalone LayerNorm or RMSNorm module because RoPE inputs are part of the forward contract.

## `fused_gated_mlp`

```python
fused_gated_mlp(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    activation: Literal["silu", "gelu"] = "silu",
) -> torch.Tensor
```

Current formula in both the Triton kernel and the reference implementation:

```text
output = activation(gate_proj(x)) * up_proj(x)
```

Input contract:

- `x`: contiguous CUDA tensor with shape `[batch, seq_len, hidden_dim]`
- `gate_weight`: contiguous CUDA tensor with shape `[intermediate_dim, hidden_dim]`
- `up_weight`: same shape as `gate_weight`
- `activation`: `"silu"` or `"gelu"`

Output:

- Shape `[batch, seq_len, intermediate_dim]`
- Same dtype as `x`

Important boundary:

- This kernel implements the gated expansion stage only.
- A full transformer MLP block still needs the down projection and residual path outside this function.

## `FusedGatedMLP`

```python
FusedGatedMLP(
    hidden_dim: int,
    intermediate_dim: int,
    activation: Literal["silu", "gelu"] = "silu",
)
```

The module owns `gate_weight` and `up_weight` and forwards to `fused_gated_mlp`.

```python
module = FusedGatedMLP(4096, 11008, activation="silu").cuda().half()
y = module(x)
```

## `fp8_gemm`

```python
fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor | None = None,
    b_scale: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor
```

Behavior:

- If `a` or `b` is floating-point, the function quantizes it internally with `quantize_fp8`.
- If `a` or `b` is already in the repository's FP8 storage format, the matching scale tensor is required.
- The current maintained runtime path uses the repository's `uint8`-based FP8 compatibility format.

Input contract:

- `a` and `b` must be contiguous CUDA tensors.
- Matrix shapes must be `[M, K]` and `[K, N]`.
- Pre-quantized inputs require scalar scale tensors on the same device.

Output:

- Shape `[M, N]`
- In normal usage, `torch.float16` or `torch.bfloat16`

Practical note:

- The validation helper accepts `torch.float32` as an output dtype, but the Triton implementation is written around half-precision output paths. Treat `float16` and `bfloat16` as the maintained choices.

## `FP8Linear`

```python
FP8Linear(in_features: int, out_features: int, bias: bool = False)
```

Behavior:

- Stores a trainable floating-point `weight` parameter.
- On the first forward pass, quantizes the weight to FP8 and caches:
  - `weight_fp8`
  - `weight_scale`
  - `weight_fp8_t` (transposed, contiguous)
- Uses `fp8_gemm` for the forward path.

Important integration caveat:

- The cached quantized weight is not automatically refreshed after weight updates.
- That makes `FP8Linear` a better fit for inference or for phases where weights are stable.

Example:

```python
import torch
from triton_ops import FP8Linear

layer = FP8Linear(4096, 4096).cuda()
x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
y = layer(x)
```
