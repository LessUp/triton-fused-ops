---
layout: default
title: "示例教程 — Triton Fused Ops"
description: "Triton Fused Ops 实用代码示例 - LLaMA、vLLM、HuggingFace 集成"
---

# 💡 示例教程

常见用例的实用代码示例。

---

## 📑 目录

1. [LLaMA 模型优化](#llama-模型优化)
2. [HuggingFace 集成](#huggingface-集成)
3. [vLLM 集成](#vllm-集成)
4. [基准测试套件](#基准测试套件)
5. [FP8 量化工作流](#fp8-量化工作流)

---

## LLaMA 模型优化

### 完整的优化 LLaMA 层

用融合算子等价替换标准 LLaMA 层：

```python
import torch
import torch.nn as nn
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP, FP8Linear

class FusedLlamaAttention(nn.Module):
    """使用 FP8 投影优化的 LLaMA Attention"""
    
    def __init__(self, hidden_dim=4096, num_heads=32, num_kv_heads=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_dim // num_heads
        
        # FP8 量化投影
        self.q_proj = FP8Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = FP8Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = FP8Linear(hidden_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = FP8Linear(hidden_dim, hidden_dim, bias=False)
        
    def forward(self, x, cos, sin, attention_mask=None):
        batch, seq_len, _ = x.shape
        
        # 投影到 Q、K、V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为 Attention 格式
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 对 Q 和 K 应用 RoPE
        #（实现取决于您的 Attention 机制）
        
        # 计算 Attention（使用 flash-attn 或标准 Attention）
        # attn_output = flash_attention(q, k, v, attention_mask)
        
        # Output 投影
        # output = self.o_proj(attn_output)
        return x  # 占位符


class FusedLlamaDecoderLayer(nn.Module):
    """完整优化的 LLaMA Decoder 层"""
    
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
        
        # 融合输入 norm 与 RoPE
        self.input_layernorm = FusedRMSNormRoPE(
            hidden_dim=hidden_dim,
            head_dim=self.head_dim,
            eps=rms_norm_eps,
        )
        
        # 融合 Post-Attention norm
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
        
        # 融合 MLP
        self.mlp = FusedGatedMLP(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            activation='silu',
        )
    
    def forward(self, x, cos, sin, attention_mask=None):
        residual = x
        
        # Pre-attention norm，融合 RoPE
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


# 使用示例
layer = FusedLlamaDecoderLayer().cuda().half()
x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
cos = torch.randn(128, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(128, 64, device='cuda', dtype=torch.float16)

output = layer(x, cos, sin)
print(f"输出形状: {output.shape}")  # [2, 128, 4096]
```

---

## HuggingFace 集成

### 修补 HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP

def patch_llama_model(model):
    """用融合算子修补 HuggingFace LLaMA 模型"""
    
    for layer in model.model.layers:
        hidden_dim = layer.input_layernorm.weight.shape[0]
        head_dim = layer.self_attn.head_dim
        
        # 替换输入 layernorm
        old_norm = layer.input_layernorm
        layer.input_layernorm = FusedRMSNormRoPE(
            hidden_dim=hidden_dim,
            head_dim=head_dim,
            eps=old_norm.variance_epsilon,
        ).cuda()
        layer.input_layernorm.weight.data = old_norm.weight.data.cuda()
        
        # 替换 post-attention norm
        old_norm = layer.post_attention_layernorm
        layer.post_attention_layernorm = FusedRMSNormRoPE(
            hidden_dim=hidden_dim,
            head_dim=head_dim,
            eps=old_norm.variance_epsilon,
        ).cuda()
        layer.post_attention_layernorm.weight.data = old_norm.weight.data.cuda()
    
    return model


# 加载并修补模型
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 应用修补
model = patch_llama_model(model)

# 生成文本
text = "人工智能的未来是"
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

print(tokenizer.decode(outputs[0]))
```

---

## vLLM 集成

### 使用融合算子的自定义 vLLM 模型

```python
# 注意：这是概念示例
# 实际集成需要 vLLM 特定的模型架构

from vllm import LLM, SamplingParams
import torch

class FusedModelAdapter:
    """与 vLLM 一起使用融合算子的适配器"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=100,
        )
    
    def load_with_fusion(self):
        """加载启用融合算子的模型"""
        # 加载自定义模型配置
        # 对模型应用融合算子
        llm = LLM(
            model=self.model_path,
            dtype="float16",
            # 融合算子的额外配置
        )
        return llm
    
    def generate(self, prompts):
        """使用优化模型生成文本"""
        llm = self.load_with_fusion()
        outputs = llm.generate(prompts, self.sampling_params)
        return outputs


# 使用
adapter = FusedModelAdapter("meta-llama/Llama-2-7b")
prompts = [
    "法国的首都是",
    "机器学习是",
]
outputs = adapter.generate(prompts)

for output in outputs:
    print(output.outputs[0].text)
```

---

## 基准测试套件

### 综合性能基准测试

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
    """Triton Fused Ops 基准测试套件"""
    
    def __init__(self, warmup=10, runs=100):
        self.warmup = warmup
        self.runs = runs
    
    def benchmark_rmsnorm_rope(self, batch, seq_len, hidden_dim, head_dim):
        """基准测试融合 RMSNorm + RoPE"""
        x = torch.randn(batch, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
        weight = torch.ones(hidden_dim, device='cuda', dtype=torch.float16)
        cos = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
        sin = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
        
        # 预热
        for _ in range(self.warmup):
            _ = fused_rmsnorm_rope(x, weight, cos, sin)
        torch.cuda.synchronize()
        
        # 基准测试
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
        """基准测试融合 Gated MLP"""
        x = torch.randn(batch, seq_len, hidden_dim, device='cuda', dtype=torch.float16)
        gate_w = torch.randn(intermediate_dim, hidden_dim, device='cuda', dtype=torch.float16)
        up_w = torch.randn(intermediate_dim, hidden_dim, device='cuda', dtype=torch.float16)
        
        # 预热
        for _ in range(self.warmup):
            _ = fused_gated_mlp(x, gate_w, up_w)
        torch.cuda.synchronize()
        
        # 基准测试
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
        """运行完整基准测试套件"""
        print("=" * 60)
        print("Triton Fused Ops 性能基准测试")
        print("=" * 60)
        
        # RMSNorm + RoPE 基准测试
        print("\n📊 RMSNorm + RoPE 融合")
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
        
        # Gated MLP 基准测试
        print("\n📊 Gated MLP 融合")
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


# 运行基准测试
benchmark = PerformanceBenchmark(warmup=10, runs=50)
benchmark.run_full_benchmark()
```

---

## FP8 量化工作流

### 完整的 FP8 量化流程

```python
import torch
from triton_ops import (
    quantize_fp8,
    dequantize_fp8,
    fp8_gemm,
    quantize_fp8_with_overflow_handling,
)

class FP8ModelQuantizer:
    """端到端 FP8 量化工作流"""
    
    def __init__(self, max_attempts=3):
        self.max_attempts = max_attempts
        self.scales = {}
    
    def quantize_tensor(self, name, tensor):
        """量化张量，带溢出处理"""
        try:
            quantized, scale = quantize_fp8_with_overflow_handling(
                tensor,
                max_attempts=self.max_attempts,
            )
            self.scales[name] = scale
            return quantized, scale
        except Exception as e:
            print(f"⚠️ 量化 {name} 失败: {e}")
            return None, None
    
    def quantize_model_weights(self, model):
        """量化模型中所有线性层的权重"""
        quantized_state = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() == 2:
                print(f"量化 {name}: {param.shape}")
                q_weight, scale = self.quantize_tensor(name, param)
                if q_weight is not None:
                    quantized_state[name] = {
                        'weight': q_weight,
                        'scale': scale,
                        'original_shape': param.shape,
                    }
        
        return quantized_state
    
    def compute_quantization_error(self, original, quantized, scale):
        """计算重建误差"""
        recovered = dequantize_fp8(quantized, scale, original.dtype)
        
        mse = torch.mean((original - recovered) ** 2).item()
        max_error = torch.max(torch.abs(original - recovered)).item()
        relative_error = (max_error / torch.max(torch.abs(original)).item()) * 100
        
        return {
            'mse': mse,
            'max_error': max_error,
            'relative_error_percent': relative_error,
        }


# 使用示例
quantizer = FP8ModelQuantizer()

# 量化线性层
linear = torch.nn.Linear(4096, 4096).cuda().half()
q_weight, scale = quantizer.quantize_tensor('test', linear.weight)

# 检查误差
if q_weight is not None:
    error = quantizer.compute_quantization_error(linear.weight, q_weight, scale)
    print(f"\n量化误差:")
    print(f"  MSE: {error['mse']:.6f}")
    print(f"  最大误差: {error['max_error']:.4f}")
    print(f"  相对误差: {error['relative_error_percent']:.2f}%")
```

---

## 🔗 相关资源

- [安装指南](./installation.md)
- [快速开始](./quickstart.md)
- [API 参考](../api/)

---

<div align="center">

**[⬆ 返回顶部](#-示例教程)** | **[← 返回文档](../)**

</div>
