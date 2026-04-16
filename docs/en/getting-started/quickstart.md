---
layout: default
title: "Quick Start — Triton Fused Ops"
description: "Get started with Triton Fused Ops in 5 minutes - your first fused kernel"
---

# 🚀 Quick Start Guide

Get up and running with Triton Fused Ops in 5 minutes.

---

## ⚡ The 3-Line Integration

Replace your existing PyTorch operations with fused kernels:

```python
import torch
from triton_ops import fused_rmsnorm_rope, fused_gated_mlp, fp8_gemm
```

---

## 🎯 Example 1: Fused RMSNorm + RoPE

### Before (Separate Operations)

```python
# Standard PyTorch - 2 kernel launches, intermediate HBM writes
x_norm = rmsnorm(x)      # Kernel 1: Load x, compute norm, write x_norm
output = rope(x_norm)    # Kernel 2: Load x_norm, compute RoPE, write output
# Total: 3 HBM reads, 2 HBM writes per element
```

### After (Fused)

```python
from triton_ops import fused_rmsnorm_rope

# Single fused kernel - no intermediate HBM writes
output = fused_rmsnorm_rope(x, weight, cos, sin)
# Total: 1 HBM read, 1 HBM write per element
```

### Complete Example

```python
import torch
from triton_ops import fused_rmsnorm_rope

# Prepare inputs
batch, seq_len, hidden_dim = 2, 128, 4096
head_dim = 64

x = torch.randn(batch, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
weight = torch.ones(hidden_dim, device='cuda', dtype=torch.float16)

# Precompute RoPE position embeddings
positions = torch.arange(seq_len, device='cuda')
freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device='cuda') / head_dim))
angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
cos = torch.cos(angles).to(torch.float16)
sin = torch.sin(angles).to(torch.float16)

# Apply fused kernel
output = fused_rmsnorm_rope(x, weight, cos, sin)
print(f"Output shape: {output.shape}")  # [2, 128, 4096]
```

---

## 🎯 Example 2: Fused Gated MLP (SwiGLU)

```python
import torch
from triton_ops import fused_gated_mlp

# LLaMA-style configuration
hidden_dim = 4096
intermediate_dim = 11008  # ~2.67x hidden_dim

x = torch.randn(2, 128, hidden_dim, device='cuda', dtype=torch.float16)
gate_w = torch.randn(intermediate_dim, hidden_dim, device='cuda', dtype=torch.float16)
up_w = torch.randn(intermediate_dim, hidden_dim, device='cuda', dtype=torch.float16)

# Single fused kernel for SwiGLU
output = fused_gated_mlp(x, gate_w, up_w, activation='silu')
print(f"Output shape: {output.shape}")  # [2, 128, 11008]
```

---

## 🎯 Example 3: FP8 Quantized GEMM

```python
import torch
from triton_ops import fp8_gemm, quantize_fp8

# Create tensors
a = torch.randn(1024, 4096, device='cuda', dtype=torch.float16)
b = torch.randn(4096, 2048, device='cuda', dtype=torch.float16)

# Method 1: Automatic quantization (recommended)
output = fp8_gemm(a, b)  # Auto-quantizes and computes

# Method 2: Pre-quantized inputs
a_fp8, a_scale = quantize_fp8(a)
b_fp8, b_scale = quantize_fp8(b)
output = fp8_gemm(a_fp8, b_fp8, a_scale, b_scale)

print(f"Output shape: {output.shape}")  # [1024, 2048]
print(f"Memory saved: 50%")
```

---

## 🏗️ Building an Optimized Transformer Layer

```python
import torch
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP, FP8Linear

class OptimizedDecoderLayer(torch.nn.Module):
    """LLaMA-style decoder with fused kernels."""
    
    def __init__(self, hidden_dim=4096, num_heads=32, intermediate_dim=11008):
        super().__init__()
        head_dim = hidden_dim // num_heads
        
        # Fused RMSNorm + RoPE
        self.input_norm = FusedRMSNormRoPE(hidden_dim, head_dim)
        
        # Fused Gated MLP
        self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim, activation='silu')
        
        # FP8 quantized projections
        self.q_proj = FP8Linear(hidden_dim, hidden_dim)
        self.k_proj = FP8Linear(hidden_dim, hidden_dim)
        self.v_proj = FP8Linear(hidden_dim, hidden_dim)
        self.o_proj = FP8Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, cos, sin):
        # Pre-norm with fused RoPE
        normed = self.input_norm(x, cos, sin)
        
        # Attention projections (FP8)
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)
        
        # Attention (use flash-attn or your implementation)
        # attn_out = flash_attn(q, k, v)
        
        # Output projection + residual
        # x = x + self.o_proj(attn_out)
        
        # MLP with residual
        x = x + self.mlp(x)
        return x

# Create model
layer = OptimizedDecoderLayer().cuda().half()
x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
cos = torch.randn(128, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(128, 64, device='cuda', dtype=torch.float16)

# Forward pass
output = layer(x, cos, sin)
print(f"Output shape: {output.shape}")
```

---

## 📊 Verifying Performance

### Benchmark Script

```python
import torch
import time
from triton_ops import fused_rmsnorm_rope

# Warm up
torch.cuda.synchronize()

# Benchmark
x = torch.randn(8, 2048, 4096, device='cuda', dtype=torch.float16)
weight = torch.ones(4096, device='cuda', dtype=torch.float16)
cos = torch.randn(2048, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(2048, 64, device='cuda', dtype=torch.float16)

# Warmup
for _ in range(10):
    _ = fused_rmsnorm_rope(x, weight, cos, sin)
torch.cuda.synchronize()

# Time it
start = time.perf_counter()
for _ in range(100):
    output = fused_rmsnorm_rope(x, weight, cos, sin)
torch.cuda.synchronize()
end = time.perf_counter()

avg_time = (end - start) / 100 * 1000
print(f"Average latency: {avg_time:.3f} ms")
```

---

## 🎓 What You Learned

- ✅ How to use the functional API (`fused_rmsnorm_rope`, `fused_gated_mlp`, `fp8_gemm`)
- ✅ How to use the Module API (`FusedRMSNormRoPE`, `FusedGatedMLP`, `FP8Linear`)
- ✅ How to build optimized transformer layers
- ✅ How to verify performance improvements

---

## 🔗 Next Steps

- [Examples](./examples.md) — More practical use cases
- [Integration Guide](../guides/integration.md) — Integrate with existing frameworks
- [API Reference](../api/kernels.md) — Complete API documentation

---

<div align="center">

**[⬆ Back to Top](#-quick-start-guide)** | **[← Back to Documentation](../)**

</div>
