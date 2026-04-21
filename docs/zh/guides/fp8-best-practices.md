---
layout: default
title: "FP8 最佳实践 — Triton Fused Ops"
description: "Triton Fused Ops FP8 量化的最佳实践"
---

# FP8 最佳实践

充分利用 FP8 量化。

---

## 📑 目录

- [何时使用 FP8](#何时使用-fp8)
- [量化策略](#量化策略)
- [精度与性能](#精度与性能)
- [常见陷阱](#常见陷阱)

---

## 何时使用 FP8

### ✅ 推荐的用例

| 场景 | 推荐 | 预期精度 |
|:---------|:---------------|:-----------------|
| **推理** | ✅ 强烈推荐 | -0.2% 到 -0.5% |
| **预训练模型** | ✅ 优秀 | <0.3% 损失 |
| **大批次推理** | ✅ 优秀 | 最小损失 |
| **边缘部署** | ✅ 内存关键 | 密切监控 |
| **训练** | ⚠️ 仔细评估 | 可能需要调整 |
| **小模型 (<1B)** | ⚠️ 先测试 | 相对影响更大 |

### ❌ 避免使用 FP8 的情况

- 最大精度至关重要（如医疗、金融）
- 模型已量化（INT8、INT4）
- 处理非常小的张量（< 1K 元素）

---

## 量化策略

### Per-Tensor vs Per-Channel

```python
import torch
from triton_ops import quantize_fp8, FP8Format, fp8_gemm, FusedRMSNormRoPE, FusedGatedMLP, FP8Linear
from triton_ops.exceptions import NumericalOverflowError

# Per-tensor 量化（默认）
tensor = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
quantized, scale = quantize_fp8(tensor)

# Per-channel 量化（用于权重）
def quantize_per_channel(weight, dim=0):
    """用逐通道缩放量化权重。"""
    scales = FP8Format.compute_scale_per_channel(weight, dim=dim)
    
    # 扩展缩放用于广播
    shape = [1] * weight.dim()
    shape[dim] = -1
    scales_expanded = scales.view(shape)
    
    # 量化
    quantized = (weight * scales_expanded).round().clamp(-448, 448).to(torch.uint8)
    return quantized, scales

# 对权重使用 per-channel
weight = torch.randn(11008, 4096, device='cuda', dtype=torch.float16)
q_weight, scales = quantize_per_channel(weight, dim=0)
```

### 动态 vs 静态缩放

```python
# 动态缩放（推荐用于激活）
def dynamic_quantize(x):
    """从输入动态计算缩放。"""
    return quantize_fp8(x)  # 从 x 计算缩放

# 静态缩放（推荐用于权重）
class StaticQuantizedLinear(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        # 预计算一次性缩放
        self.register_buffer('weight', weight)
        self.register_buffer('scale', FP8Format.compute_scale(weight))
        self.register_buffer('weight_fp8', None)
    
    def forward(self, x):
        # 一次性量化权重
        if self.weight_fp8 is None:
            self.weight_fp8 = (self.weight * self.scale).round().clamp(-448, 448).to(torch.uint8)
        
        # 动态量化激活
        x_fp8, x_scale = quantize_fp8(x)
        
        # 用 fp8_gemm 计算
        return fp8_gemm(x_fp8, self.weight_fp8.t(), x_scale, self.scale)
```

---

## 精度与性能

### 按层类型的 FP8 影响

| 层类型 | FP8 影响 | 推荐 |
|:-----------|:-----------|:---------------|
| **Embedding** | 高 | 保持 FP16/BF16 |
| **Attention Q/K/V** | 低 | 可安全量化 |
| **Attention Output** | 中 | 仔细测试 |
| **MLP Gate/Up** | 低 | 可安全量化 |
| **LayerNorm/RMSNorm** | 高 | 保持 FP32/FP16 |
| **LM Head** | 高 | 保持 FP16/BF16 |

### 混合精度策略

```python
class MixedPrecisionTransformerLayer(torch.nn.Module):
    """在安全的地方用 FP8，在需要的地方用 FP16。"""
    
    def __init__(self, hidden_dim, num_heads, intermediate_dim):
        super().__init__()
        
        # 量化投影
        self.q_proj = FP8Linear(hidden_dim, hidden_dim)
        self.k_proj = FP8Linear(hidden_dim, hidden_dim)
        self.v_proj = FP8Linear(hidden_dim, hidden_dim)
        
        # 为精度保持输出投影为 FP16
        self.o_proj = torch.nn.Linear(hidden_dim, hidden_dim, dtype=torch.float16)
        
        # 保持 norm 为 FP16
        self.norm = FusedRMSNormRoPE(hidden_dim, hidden_dim // num_heads)
        
        # 量化 MLP
        self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim)
    
    def forward(self, x, cos, sin):
        # FP8 量化 attention
        normed = self.norm(x, cos, sin)
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)
        
        # Attention 计算...
        
        # FP16 输出投影
        out = self.o_proj(attn_output)
        
        # FP8 MLP
        x = x + out
        x = x + self.mlp(x)
        
        return x
```

### 精度校准

```python
def calibrate_fp8_layers(model, calibration_data):
    """校准 FP8 量化范围。"""
    
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
    
    # 注册 hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, FP8Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # 运行校准
    with torch.no_grad():
        for batch in calibration_data:
            _ = model(batch)
    
    # 移除 hooks
    for hook in hooks:
        hook.remove()
    
    # 计算最优缩放
    scales = {}
    for name, stats in activation_stats.items():
        max_vals = [s['abs_max'] for s in stats]
        global_max = max(max_vals)
        scales[name] = 448.0 / global_max
    
    return scales
```

---

## 常见陷阱

### 1. 溢出处理

```python
# ❌ 不好：忽略溢出
q, s = quantize_fp8(tensor * 1000)  # 可能溢出！

# ✅ 好：使用溢出处理
from triton_ops import quantize_fp8_with_overflow_handling

try:
    q, s = quantize_fp8_with_overflow_handling(tensor * 1000, max_attempts=3)
except NumericalOverflowError:
    # 回退到 FP16
    pass
```

### 2. 缩放不匹配

```python
# ❌ 不好：缩放不匹配
q1, s1 = quantize_fp8(a)
q2, s2 = quantize_fp8(b)
# 对 q2 用 s1 - 错误！
result = fp8_gemm(q1, q2, s1, s1)  # Bug！

# ✅ 好：用正确的缩放
result = fp8_gemm(q1, q2, s1, s2)  # 正确
```

### 3. 忘记同步

```python
# ❌ 不好：没有同步的计时
start = time.time()
result = fp8_gemm(a, b)
print(f"时间: {time.time() - start}")  # 错误！

# ✅ 好：正确的同步
torch.cuda.synchronize()
start = time.time()
result = fp8_gemm(a, b)
torch.cuda.synchronize()
print(f"时间: {time.time() - start}")  # 正确
```

### 4. 过早量化

```python
# ❌ 不好：在 layer norm 之前量化
x_fp8, _ = quantize_fp8(x)
x_norm = rmsnorm(dequantize_fp8(x_fp8, scale))  # 精度损失！

# ✅ 好：用更高精度做 norm
x_norm = rmsnorm(x)  # 保持精度
q, s = quantize_fp8(x_norm)  # 然后量化
```

---

## 测试精度

```python
def test_fp8_accuracy(model, test_data, tolerance=0.01):
    """测试 FP8 量化的模型精度。"""
    
    # 基准：FP16
    model_fp16 = model.clone().half()
    acc_fp16 = evaluate(model_fp16, test_data)
    
    # 测试：FP8
    model_fp8 = convert_to_fp8(model)
    acc_fp8 = evaluate(model_fp8, test_data)
    
    # 比较
    accuracy_drop = acc_fp16 - acc_fp8
    print(f"FP16 精度: {acc_fp16:.4f}")
    print(f"FP8 精度:  {acc_fp8:.4f}")
    print(f"精度下降: {accuracy_drop:.4f} ({accuracy_drop/acc_fp16*100:.2f}%)")
    
    assert accuracy_drop < tolerance, f"精度下降 {accuracy_drop} 超过容差"
    
    return acc_fp8

def evaluate(model, data):
    """在测试数据上评估模型。"""
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

**[⬆ 返回顶部](#fp8-最佳实践)** | **[← 返回指南](../)**

</div>
