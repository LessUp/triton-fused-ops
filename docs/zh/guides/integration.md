---
layout: default
title: "集成指南 — Triton Fused Ops"
description: "与 HuggingFace、PyTorch 和 vLLM 的集成指南"
---

# 集成指南

将 Triton Fused Ops 与流行框架集成。

---

## 📑 目录

- [HuggingFace Transformers](#huggingface-transformers)
- [PyTorch 模型](#pytorch-模型)
- [vLLM](#vllm)
- [自定义训练循环](#自定义训练循环)

---

## HuggingFace Transformers

### 修补 LLaMA 模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP

def patch_llama_for_fusion(model):
    """用融合算子修补 HuggingFace LLaMA 模型。"""
    
    for layer in model.model.layers:
        hidden_dim = layer.input_layernorm.weight.shape[0]
        head_dim = layer.self_attn.head_dim
        
        # 用融合版本替换输入 layernorm
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


# 加载模型
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 应用融合修补
model = patch_llama_for_fusion(model)

# 正常使用
text = "人工智能的未来是"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### 与训练集成

```python
from transformers import Trainer, TrainingArguments

# 修补后
model = patch_llama_for_fusion(model)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 训练时将使用融合算子
trainer.train()
```

---

## PyTorch 模型

### 原生 PyTorch 集成

```python
import torch
import torch.nn as nn
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP, FP8Linear

class MyTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, intermediate_dim):
        super().__init__()
        head_dim = hidden_dim // num_heads
        
        # 使用融合模块
        self.norm = FusedRMSNormRoPE(hidden_dim, head_dim)
        self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim)
        
        # 可选：FP8 量化投影
        self.proj = FP8Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, cos, sin):
        x = self.norm(x, cos, sin)
        x = self.mlp(x)
        return x

# 在模型中使用
layer = MyTransformerLayer(4096, 32, 11008).cuda().half()
x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
cos = torch.randn(128, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(128, 64, device='cuda', dtype=torch.float16)

output = layer(x, cos, sin)
```

### torch.compile 兼容性

```python
import torch

# 融合算子与 torch.compile 兼容
model = MyTransformerLayer(4096, 32, 11008).cuda()
compiled_model = torch.compile(model)

# 使用编译后的模型
output = compiled_model(x, cos, sin)
```

---

## vLLM

### 使用融合算子的自定义模型

```python
# 在 vLLM 模型定义中
from vllm.model_executor.layers.linear import LinearMethodBase
from triton_ops import FP8Linear

class FusedvLLMModel:
    """使用融合算子的 vLLM 模型示例。"""
    
    def __init__(self, config):
        # 使用 FP8 线性层进行量化
        self.qkv_proj = FP8Linear(
            config.hidden_size,
            3 * config.hidden_size,
            bias=False,
        )
        self.o_proj = FP8Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )
```

### 推理配置

```python
# vLLM 推理，使用优化算子
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b",
    dtype="float16",
    # 启用优化注意力（如果支持）
    attention_impl="flash_attn",
)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(["你好，世界！"], sampling_params)
```

---

## 自定义训练循环

### 混合精度训练

```python
import torch
from torch.cuda.amp import autocast, GradScaler
from triton_ops import fused_gated_mlp, fp8_gemm

# 设置
model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters())
scaler = GradScaler()

# 训练循环
for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast(dtype=torch.float16):
        # 融合算子自动与 autocast 兼容
        output = model(batch)
        loss = compute_loss(output)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 梯度检查点

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, intermediate_dim):
        super().__init__()
        self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim)
    
    def forward(self, x):
        # 使用检查点节省内存
        return checkpoint(self.mlp, x)
```

---

<div align="center">

**[⬆ 返回顶部](#集成指南)** | **[← 返回指南](../)**

</div>
