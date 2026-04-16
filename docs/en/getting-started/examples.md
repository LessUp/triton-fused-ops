---
layout: default
title: "Examples — Triton Fused Ops"
description: "Practical code examples for Triton Fused Ops - LLaMA, vLLM, HuggingFace integration"
---

# 💡 Examples

Practical code examples for common use cases.

---

## 📑 Table of Contents

1. [LLaMA Model Optimization](#llama-model-optimization)
2. [HuggingFace Integration](#huggingface-integration)
3. [vLLM Integration](#vllm-integration)
4. [Benchmark Suite](#benchmark-suite)
5. [FP8 Quantization Workflow](#fp8-quantization-workflow)

---

## LLaMA Model Optimization

### Complete Optimized LLaMA Layer

Replace standard LLaMA layers with fused equivalents:

```python
import torch
import torch.nn as nn
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP, FP8Linear

class FusedLlamaAttention(nn.Module):
    """Optimized LLaMA attention with FP8 projections."""
    
    def __init__(self, hidden_dim=4096, num_heads=32, num_kv_heads=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_dim // num_heads
        
        # FP8 quantized projections
        self.q_proj = FP8Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = FP8Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = FP8Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = FP8Linear(hidden_dim, hidden_dim, bias=False)
        
    def forward(self, x, cos, sin, attention_mask=None):
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for attention
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        # (Implementation depends on your attention mechanism)
        
        # Compute attention (use flash-attn or standard attention)
        # attn_output = flash_attention(q, k, v, attention_mask)
        
        # Output projection
        # output = self.o_proj(attn_output)
        return x  # Placeholder


class FusedLlamaDecoderLayer(nn.Module):
    """Complete optimized LLaMA decoder layer."""
    
    def __init__(
        self,
        hidden_dim=4096,
        num_heads=32,
        num_kv_heads=None,
        intermediate_dim=11008,
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        # Fused input norm with RoPE
        self.input_layernorm = FusedRMSNormRoPE(
            hidden_dim=hidden_dim,
            head_dim=self.head_dim,
            eps=rms_norm_eps,
        )
        
        # Fused post-attention norm
        self.post_attention_layernorm = FusedRMSNormRoPE(
            hidden_dim=hidden_dim,
            head_dim=self.head_dim,
            eps=rms_norm_eps,
        )
        
        # Attention
        self.self_attn = FusedLlamaAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        
        # Fused MLP
        self.mlp = FusedGatedMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            activation='silu',
        )
    
    def forward(self, x, cos, sin, attention_mask=None):
        residual = x
        
        # Pre-attention norm with fused RoPE
        hidden_states = self.input_layernorm(x, cos, sin)
        
        # Self attention
        hidden_states = self.self_attn(
            hidden_states, cos, sin, attention_mask
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


# Usage example
layer = FusedLlamaDecoderLayer().cuda().half()
x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
cos = torch.randn(128, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(128, 64, device='cuda', dtype=torch.float16)

output = layer(x, cos, sin)
print(f"Output shape: {output.shape}")  # [2, 128, 4096]
```

---

## HuggingFace Integration

### Patch HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP

def patch_llama_model(model):
    """Patch a HuggingFace LLaMA model with fused kernels."""
    
    for layer in model.model.layers:
        hidden_dim = layer.input_layernorm.weight.shape[0]
        head_dim = layer.self_attn.head_dim
        
        # Replace input layernorm
        old_norm = layer.input_layernorm
        layer.input_layernorm = FusedRMSNormRoPE(
            hidden_dim=hidden_dim,
            head_dim=head_dim,
            eps=old_norm.variance_epsilon,
        ).cuda()
        layer.input_layernorm.weight.data = old_norm.weight.data.cuda()
        
        # Replace post-attention norm
        old_norm = layer.post_attention_layernorm
        layer.post_attention_layernorm = FusedRMSNormRoPE(
            hidden_dim=hidden_dim,
            head_dim=head_dim,
            eps=old_norm.variance_epsilon,
        ).cuda()
        layer.post_attention_layernorm.weight.data = old_norm.weight.data.cuda()
    
    return model


# Load and patch model
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Apply patches
model = patch_llama_model(model)

# Generate
text = "The future of AI is"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0]))
```

---

## vLLM Integration

### Custom vLLM Model with Fused Kernels

```python
# Note: This is a conceptual example
# Actual integration requires vLLM's specific model architecture

from vllm import LLM, SamplingParams
import torch

class FusedModelAdapter:
    """Adapter to use fused kernels with vLLM."""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=100,
        )
    
    def load_with_fusion(self):
        """Load model with fused kernels enabled."""
        # Load custom model configuration
        # Apply fused kernels to the model
        llm = LLM(
            model=self.model_path,
            dtype="float16",
            # Additional configuration for fused kernels
        )
        return llm
    
    def generate(self, prompts):
        """Generate text with optimized model."""
        llm = self.load_with_fusion()
        outputs = llm.generate(prompts, self.sampling_params)
        return outputs


# Usage
adapter = FusedModelAdapter("meta-llama/Llama-2-7b")
prompts = [
    "The capital of France is",
    "Machine learning is",
]
outputs = adapter.generate(prompts)

for output in outputs:
    print(output.outputs[0].text)
```

---

## Benchmark Suite

### Comprehensive Performance Benchmark

```python
import torch
import time
import statistics
from triton_ops import (
    fused_rmsnorm_rope,
    fused_gated_mlp,
    fp8_gemm,
    FusedRMSNormRoPE,
    FusedGatedMLP,
    FP8Linear,
)

class PerformanceBenchmark:
    """Benchmark suite for Triton Fused Ops."""
    
    def __init__(self, warmup=10, runs=100):
        self.warmup = warmup
        self.runs = runs
    
    def benchmark_rmsnorm_rope(self, batch, seq_len, hidden_dim, head_dim):
        """Benchmark fused RMSNorm + RoPE."""
        x = torch.randn(batch, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
        weight = torch.ones(hidden_dim, device='cuda', dtype=torch.float16)
        cos = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
        sin = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
        
        # Warmup
        for _ in range(self.warmup):
            _ = fused_rmsnorm_rope(x, weight, cos, sin)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(self.runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            output = fused_rmsnorm_rope(x, weight, cos, sin)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        return {
            'mean_ms': statistics.mean(times),
            'std_ms': statistics.stdev(times),
            'min_ms': min(times),
            'max_ms': max(times),
        }
    
    def benchmark_gated_mlp(self, batch, seq_len, hidden_dim, intermediate_dim):
        """Benchmark fused Gated MLP."""
        x = torch.randn(batch, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
        gate_w = torch.randn(intermediate_dim, hidden_dim, device='cuda', dtype=torch.float16)
        up_w = torch.randn(intermediate_dim, hidden_dim, device='cuda', dtype=torch.float16)
        
        # Warmup
        for _ in range(self.warmup):
            _ = fused_gated_mlp(x, gate_w, up_w)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(self.runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            output = fused_gated_mlp(x, gate_w, up_w)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        
        return {
            'mean_ms': statistics.mean(times),
            'std_ms': statistics.stdev(times),
            'min_ms': min(times),
            'max_ms': max(times),
        }
    
    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("=" * 60)
        print("Triton Fused Ops Performance Benchmark")
        print("=" * 60)
        
        # RMSNorm + RoPE benchmarks
        print("\n📊 RMSNorm + RoPE Fusion")
        print("-" * 60)
        configs = [
            (1, 2048, 4096, 128),
            (4, 2048, 4096, 128),
            (8, 2048, 4096, 128),
            (16, 4096, 4096, 128),
        ]
        for batch, seq_len, hidden_dim, head_dim in configs:
            result = self.benchmark_rmsnorm_rope(batch, seq_len, hidden_dim, head_dim)
            print(f"Batch={batch:2d}, Seq={seq_len:4d}: {result['mean_ms']:.3f} ± {result['std_ms']:.3f} ms")
        
        # Gated MLP benchmarks
        print("\n📊 Gated MLP Fusion")
        print("-" * 60)
        configs = [
            (1, 2048, 4096, 11008),
            (4, 2048, 4096, 11008),
            (8, 2048, 4096, 11008),
        ]
        for batch, seq_len, hidden_dim, intermediate_dim in configs:
            result = self.benchmark_gated_mlp(batch, seq_len, hidden_dim, intermediate_dim)
            print(f"Batch={batch:2d}, Seq={seq_len:4d}: {result['mean_ms']:.3f} ± {result['std_ms']:.3f} ms")
        
        print("\n" + "=" * 60)


# Run benchmark
benchmark = PerformanceBenchmark(warmup=10, runs=50)
benchmark.run_full_benchmark()
```

---

## FP8 Quantization Workflow

### Complete FP8 Quantization Pipeline

```python
import torch
from triton_ops import (
    quantize_fp8,
    dequantize_fp8,
    fp8_gemm,
    quantize_fp8_with_overflow_handling,
)

class FP8ModelQuantizer:
    """End-to-end FP8 quantization workflow."""
    
    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts
        self.scales = {}
    
    def quantize_tensor(self, name, tensor):
        """Quantize a tensor with overflow handling."""
        try:
            quantized, scale = quantize_fp8_with_overflow_handling(
                tensor,
                max_attempts=self.max_attempts,
            )
            self.scales[name] = scale
            return quantized, scale
        except Exception as e:
            print(f"⚠️ Failed to quantize {name}: {e}")
            return None, None
    
    def quantize_model_weights(self, model):
        """Quantize all linear layer weights in a model."""
        quantized_state = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() == 2:
                print(f"Quantizing {name}: {param.shape}")
                q_weight, scale = self.quantize_tensor(name, param)
                if q_weight is not None:
                    quantized_state[name] = {
                        'weight': q_weight,
                        'scale': scale,
                        'original_shape': param.shape,
                    }
        
        return quantized_state
    
    def compute_quantization_error(self, original, quantized, scale):
        """Compute reconstruction error."""
        recovered = dequantize_fp8(quantized, scale, original.dtype)
        
        mse = torch.mean((original - recovered) ** 2).item()
        max_error = torch.max(torch.abs(original - recovered)).item()
        relative_error = (max_error / torch.max(torch.abs(original)).item()) * 100
        
        return {
            'mse': mse,
            'max_error': max_error,
            'relative_error_percent': relative_error,
        }


# Usage example
quantizer = FP8ModelQuantizer()

# Quantize a linear layer
linear = torch.nn.Linear(4096, 4096).cuda().half()
q_weight, scale = quantizer.quantize_tensor('test', linear.weight)

# Check error
if q_weight is not None:
    error = quantizer.compute_quantization_error(linear.weight, q_weight, scale)
    print(f"\nQuantization Error:")
    print(f"  MSE: {error['mse']:.6f}")
    print(f"  Max Error: {error['max_error']:.4f}")
    print(f"  Relative Error: {error['relative_error_percent']:.2f}%")
```

---

## 🔗 Related Resources

- [Installation Guide](./installation.md)
- [Quick Start](./quickstart.md)
- [API Reference](../api/)

---

<div align="center">

**[⬆ Back to Top](#-examples)** | **[← Back to Documentation](../)**

</div>
