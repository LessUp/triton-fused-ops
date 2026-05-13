---
title: 示例教程
description: "基于当前 Triton Fused Ops API 的可复用代码模式"
---

# 示例教程

这里的示例尽量贴合仓库当前真实导出的接口与行为。

## Decoder block 骨架

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

这说明本仓库更适合被当成“优化原语集合”使用，而不是完整的 Attention / Transformer 实现。

## 用数据模型生成测试输入

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

`models.py` 中的 dataclass 很适合用于测试、示例和 benchmark 脚手架。

## 使用 `BenchmarkSuite` 做基准测试

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

## 对自定义 kernel wrapper 做自动调优

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

`TritonAutoTuner` 期望被调优的函数能够接受搜索出来的配置参数作为关键字参数。

## FP8 溢出辅助函数

带溢出处理的 helper 位于 kernel 模块中，而不是根包导出列表里：

```python
import torch
from triton_ops.kernels.fp8_quantize import quantize_fp8_with_overflow_handling

x = torch.full((1024,), 1000.0, device="cuda", dtype=torch.float16)
q, scale = quantize_fp8_with_overflow_handling(x, max_attempts=3)
print(q.dtype, scale.item())
```

## 下一步

- [集成指南](/zh/guides/integration)
- [基准测试 API](/zh/api/benchmark)
- [数据模型 API](/zh/api/models)
