---
layout: default
title: Examples
parent: Getting Started
grand_parent: Documentation
nav_order: 3
description: "Reusable code patterns built on the current Triton Fused Ops API"
---

# Examples

These examples stay close to what the repository actually exports today.

## Decoder-block skeleton

```python
import torch
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP, FP8Linear

class DecoderBlock(torch.nn.Module):
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

This repository does not ship a full attention implementation. Use your existing attention path around the fused normalization, projection, and MLP pieces.

## Data models for test input generation

```python
import torch
from triton_ops import RMSNormRoPEInput

spec = RMSNormRoPEInput.from_shapes(
    batch_size=2,
    seq_len=128,
    hidden_dim=4096,
    head_dim=64,
    dtype=torch.float16,
    device="cuda",
)

x = spec.x.create_tensor()
weight = spec.weight.create_tensor(fill_value=1.0)
cos = spec.cos.create_tensor()
sin = spec.sin.create_tensor()
```

The `models.py` dataclasses are useful for tests, examples, and benchmark scaffolding.

## Benchmarking with `BenchmarkSuite`

```python
import torch
from triton_ops import BenchmarkSuite
from triton_ops.kernels.rmsnorm_rope import fused_rmsnorm_rope_reference
from triton_ops import fused_rmsnorm_rope

suite = BenchmarkSuite(warmup_runs=5, benchmark_runs=20)

x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
weight = torch.ones(4096, device="cuda", dtype=torch.float16)
cos = torch.randn(128, 64, device="cuda", dtype=torch.float16)
sin = torch.randn(128, 64, device="cuda", dtype=torch.float16)

result = suite.benchmark_kernel(
    fused_rmsnorm_rope,
    fused_rmsnorm_rope_reference,
    "fused_rmsnorm_rope",
    (2, 128, 4096),
    x,
    weight,
    cos,
    sin,
)

print(result.metrics.latency_ms)
print(suite.generate_report())
```

## Auto-tuning a custom kernel wrapper

```python
import torch
from triton_ops import TritonAutoTuner

def dummy_kernel(x, BLOCK_SIZE=64, num_warps=4):
    return x * 2

tuner = TritonAutoTuner(
    kernel_fn=dummy_kernel,
    config_space={
        "BLOCK_SIZE": [64, 128],
        "num_warps": [4, 8],
    },
    warmup_runs=2,
    benchmark_runs=10,
)

x = torch.randn(1024, device="cuda")
result = tuner.tune(x, problem_size=(1024,), device="cuda:0", kernel_type="dummy")
print(result.best_config)
```

`TritonAutoTuner` expects a callable that accepts the searched configuration values as keyword arguments.

## FP8 overflow helper

The overflow-handling helper lives in the kernel module rather than the root package export list:

```python
import torch
from triton_ops.kernels.fp8_quantize import quantize_fp8_with_overflow_handling

x = torch.full((1024,), 1000.0, device="cuda", dtype=torch.float16)
q, scale = quantize_fp8_with_overflow_handling(x, max_attempts=3)
print(q.dtype, scale.item())
```

## Next

- [Integration Guide]({{ '/docs/en/guides/integration/' | relative_url }})
- [Benchmark API]({{ '/docs/en/api/benchmark/' | relative_url }})
- [Models API]({{ '/docs/en/api/models/' | relative_url }})
