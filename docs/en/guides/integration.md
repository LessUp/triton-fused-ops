---
title: Integration Guide
description: "How to integrate the shipped kernels into larger model code"
---

# Integration Guide

This page focuses on realistic integration boundaries for the code that exists in the repository today.

## Choose the right integration level

### Functional API

Use the functional entry points when:

- your model already owns the weights and caches,
- you want minimal wrapper overhead,
- you are patching only a few hot-path calls.

Relevant functions:

- `fused_rmsnorm_rope`
- `fused_gated_mlp`
- `fp8_gemm`
- `quantize_fp8` / `dequantize_fp8`

### Module wrappers

Use the module wrappers when:

- you want `nn.Module`-style composition,
- you prefer weights to live inside the module,
- you are building inference-oriented blocks from the repository primitives.

Relevant modules:

- `FusedRMSNormRoPE`
- `FusedGatedMLP`
- `FP8Linear`

## Important runtime boundaries

### `FusedRMSNormRoPE` is not a plain norm layer

Its forward signature requires RoPE inputs:

```python
out = module(x, cos, sin)
```

That means it fits best where your model already has access to position-cache tensors.

### `FusedGatedMLP` covers only the gated expansion stage

The repository module returns the intermediate gated activation output. A full decoder FFN still needs the down projection and surrounding residual logic outside the module.

### `FP8Linear` is best treated as an inference-oriented building block

The module quantizes and caches its weights on first forward. If the floating weight parameter changes later, the cached FP8 buffers are not automatically refreshed.

## Decoder-block sketch

```python
import torch
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP, FP8Linear

class DecoderSlice(torch.nn.Module):
    def __init__(self, hidden_dim=4096, num_heads=32, intermediate_dim=11008):
        super().__init__()
        head_dim = hidden_dim // num_heads
        self.norm = FusedRMSNormRoPE(hidden_dim, head_dim)
        self.q_proj = FP8Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = FP8Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = FP8Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim, activation="silu")

    def forward(self, x, cos, sin):
        normed = self.norm(x, cos, sin)
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)
        mlp_out = self.mlp(normed)
        return q, k, v, mlp_out
```

## HuggingFace or custom-model patching

The repository does not ship an official HuggingFace or vLLM adapter. The practical pattern is:

1. identify the exact model submodule that owns the tensors you need,
2. replace only the hot-path norm / projection / MLP pieces,
3. keep the model's original attention implementation unless you also own that path,
4. validate numerics against the pre-patch model on representative inputs.

In other words, use this repository as a set of optimized primitives, not as a framework-level drop-in integration package.

## Validation checklist before wiring into a model

- keep tensors on CUDA,
- keep inputs contiguous,
- check RoPE cache shape carefully,
- verify hidden dimension divisibility by `head_dim`,
- benchmark with warmup and synchronization,
- compare outputs against the unfused baseline before rollout.
