---
layout: default
title: "FP8 Best Practices — Triton Fused Ops"
description: "Best practices for FP8 quantization in Triton Fused Ops"
---

# FP8 Best Practices

Get the most out of FP8 quantization.

---

## 📑 Table of Contents

- [When to Use FP8](#when-to-use-fp8)
- [Quantization Strategy](#quantization-strategy)
- [Accuracy vs Performance](#accuracy-vs-performance)
- [Common Pitfalls](#common-pitfalls)

---

## When to Use FP8

### ✅ Recommended Use Cases

| Scenario | Recommendation | Expected Accuracy |
|:---------|:---------------|:-----------------|
| **Inference** | ✅ Strongly recommended | -0.2% to -0.5% |
| **Pre-trained models** | ✅ Excellent | <0.3% loss |
| **Large batch inference** | ✅ Excellent | Minimal loss |
| **Edge deployment** | ✅ Memory critical | Monitor closely |
| **Training** | ⚠️ Careful evaluation | May need adjustments |
| **Small models (<1B)** | ⚠️ Test first | Higher relative impact |

### ❌ Avoid FP8 When

- Maximum accuracy is critical (e.g., medical, financial)
- Model is already quantized (INT8, INT4)
- Working with very small tensors (< 1K elements)

---

## Quantization Strategy

### Per-Tensor vs Per-Channel

```python
import torch
from triton_ops import quantize_fp8, FP8Format

# Per-tensor quantization (default)
tensor = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
quantized, scale = quantize_fp8(tensor)

# Per-channel quantization (for weights)
def quantize_per_channel(weight, dim=0):
    """Quantize weight with per-channel scales."""
    scales = FP8Format.compute_scale_per_channel(weight, dim=dim)
    
    # Expand scales for broadcasting
    shape = [1] * weight.dim()
    shape[dim] = -1
    scales_expanded = scales.view(shape)
    
    # Quantize
    quantized = (weight * scales_expanded).round().clamp(-448, 448).to(torch.uint8)
    return quantized, scales

# Use per-channel for weights
weight = torch.randn(11008, 4096, device='cuda', dtype=torch.float16)
q_weight, scales = quantize_per_channel(weight, dim=0)
```

### Dynamic vs Static Scaling

```python
# Dynamic scaling (recommended for activations)
def dynamic_quantize(x):
    """Compute scale dynamically per input."""
    return quantize_fp8(x)  # Scale computed from x

# Static scaling (recommended for weights)
class StaticQuantizedLinear(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        # Pre-compute scale once
        self.register_buffer('weight', weight)
        self.register_buffer('scale', FP8Format.compute_scale(weight))
        self.register_buffer('weight_fp8', None)
    
    def forward(self, x):
        # Quantize weight once
        if self.weight_fp8 is None:
            self.weight_fp8 = (self.weight * self.scale).round().clamp(-448, 448).to(torch.uint8)
        
        # Quantize activation dynamically
        x_fp8, x_scale = quantize_fp8(x)
        
        # Compute with fp8_gemm
        return fp8_gemm(x_fp8, self.weight_fp8.t(), x_scale, self.scale)
```

---

## Accuracy vs Performance

### Accuracy Impact by Layer Type

| Layer Type | FP8 Impact | Recommendation |
|:-----------|:-----------|:---------------|
| **Embedding** | High | Keep FP16/BF16 |
| **Attention Q/K/V** | Low | Safe to quantize |
| **Attention Output** | Medium | Test carefully |
| **MLP Gate/Up** | Low | Safe to quantize |
| **LayerNorm/RMSNorm** | High | Keep FP32/FP16 |
| **LM Head** | High | Keep FP16/BF16 |

### Mixed Precision Strategy

```python
class MixedPrecisionTransformerLayer(torch.nn.Module):
    """Use FP8 where safe, FP16 where needed."""
    
    def __init__(self, hidden_dim, num_heads, intermediate_dim):
        super().__init__()
        
        # Quantized projections
        self.q_proj = FP8Linear(hidden_dim, hidden_dim)
        self.k_proj = FP8Linear(hidden_dim, hidden_dim)
        self.v_proj = FP8Linear(hidden_dim, hidden_dim)
        
        # Keep output projection in FP16 for accuracy
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, dtype=torch.float16)
        
        # Keep norms in FP16
        self.norm = FusedRMSNormRoPE(hidden_dim, hidden_dim // num_heads)
        
        # Quantized MLP
        self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim)
    
    def forward(self, x, cos, sin):
        # FP8 quantized attention
        normed = self.norm(x, cos, sin)
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)
        
        # Attention computation...
        
        # FP16 output projection
        out = self.o_proj(attn_output)
        
        # FP8 MLP
        x = x + out
        x = x + self.mlp(x)
        
        return x
```

### Calibration for Accuracy

```python
def calibrate_fp8_layers(model, calibration_data):
    """Calibrate FP8 quantization ranges."""
    
    model.eval()
    activation_stats = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                if name not in activation_stats:
                    activation_stats[name] = []
                activation_stats[name].append({
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'abs_max': output.abs().max().item(),
                })
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, FP8Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Run calibration
    with torch.no_grad():
        for batch in calibration_data:
            _ = model(batch)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Compute optimal scales
    scales = {}
    for name, stats in activation_stats.items():
        max_vals = [s['abs_max'] for s in stats]
        global_max = max(max_vals)
        scales[name] = 448.0 / global_max
    
    return scales
```

---

## Common Pitfalls

### 1. Overflow Handling

```python
# ❌ Bad: Ignoring overflow
q, s = quantize_fp8(tensor * 1000)  # May overflow!

# ✅ Good: Use overflow handling
from triton_ops import quantize_fp8_with_overflow_handling

try:
    q, s = quantize_fp8_with_overflow_handling(tensor * 1000, max_attempts=3)
except NumericalOverflowError:
    # Fallback to FP16
    pass
```

### 2. Scale Mismatch

```python
# ❌ Bad: Mismatched scales
q1, s1 = quantize_fp8(a)
q2, s2 = quantize_fp8(b)
# Use s1 for q2 - wrong!
result = fp8_gemm(q1, q2, s1, s1)  # Bug!

# ✅ Good: Use correct scales
result = fp8_gemm(q1, q2, s1, s2)  # Correct
```

### 3. Forgetting to Synchronize

```python
# ❌ Bad: Timing without sync
start = time.time()
result = fp8_gemm(a, b)
print(f"Time: {time.time() - start}")  # Wrong!

# ✅ Good: Proper synchronization
torch.cuda.synchronize()
start = time.time()
result = fp8_gemm(a, b)
torch.cuda.synchronize()
print(f"Time: {time.time() - start}")  # Correct
```

### 4. Quantizing Too Early

```python
# ❌ Bad: Quantize before layer norm
x_fp8, _ = quantize_fp8(x)
x_norm = rmsnorm(dequantize_fp8(x_fp8, scale))  # Accuracy loss!

# ✅ Good: Norm in higher precision
x_norm = rmsnorm(x)  # Keep precision
q, s = quantize_fp8(x_norm)  # Then quantize
```

---

## Testing Accuracy

```python
def test_fp8_accuracy(model, test_data, tolerance=0.01):
    """Test model accuracy with FP8 quantization."""
    
    # Baseline: FP16
    model_fp16 = model.clone().half()
    acc_fp16 = evaluate(model_fp16, test_data)
    
    # Test: FP8
    model_fp8 = convert_to_fp8(model)
    acc_fp8 = evaluate(model_fp8, test_data)
    
    # Compare
    accuracy_drop = acc_fp16 - acc_fp8
    print(f"FP16 Accuracy: {acc_fp16:.4f}")
    print(f"FP8 Accuracy:  {acc_fp8:.4f}")
    print(f"Accuracy Drop: {accuracy_drop:.4f} ({accuracy_drop/acc_fp16*100:.2f}%)")
    
    assert accuracy_drop < tolerance, f"Accuracy drop {accuracy_drop} exceeds tolerance"
    
    return acc_fp8

def evaluate(model, data):
    """Evaluate model on test data."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in data:
            pred = model(x).argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total
```

---

<div align="center">

**[⬆ Back to Top](#fp8-best-practices)** | **[← Back to Guides](../)**

</div>
