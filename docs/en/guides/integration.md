---
layout: default
title: "Integration Guide — Triton Fused Ops"
description: "Integration guide for HuggingFace, PyTorch, and vLLM"
---

# Integration Guide

Integrate Triton Fused Ops with popular frameworks.

---

## 📑 Table of Contents

- [HuggingFace Transformers](#huggingface-transformers)
- [PyTorch Models](#pytorch-models)
- [vLLM](#vllm)
- [Custom Training Loops](#custom-training-loops)

---

## HuggingFace Transformers

### Patching LLaMA Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP

def patch_llama_for_fusion(model):
    """Patch a HuggingFace LLaMA model with fused kernels."""
    
    for layer in model.model.layers:
        hidden_dim = layer.input_layernorm.weight.shape[0]
        head_dim = layer.self_attn.head_dim
        
        # Replace input layernorm with fused version
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


# Load model
model_id = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Apply fusion patches
model = patch_llama_for_fusion(model)

# Use as normal
text = "The future of AI is"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### Integration with Training

```python
from transformers import Trainer, TrainingArguments

# After patching
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

# Fused kernels will be used during training
trainer.train()
```

---

## PyTorch Models

### Native PyTorch Integration

```python
import torch
import torch.nn as nn
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP, FP8Linear

class MyTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, intermediate_dim):
        super().__init__()
        head_dim = hidden_dim // num_heads
        
        # Use fused modules
        self.norm = FusedRMSNormRoPE(hidden_dim, head_dim)
        self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim)
        
        # Optional: FP8 quantized projections
        self.proj = FP8Linear(hidden_dim, hidden_dim)
    
    def forward(self, x, cos, sin):
        x = self.norm(x, cos, sin)
        x = self.mlp(x)
        return x

# Use in your model
layer = MyTransformerLayer(4096, 32, 11008).cuda().half()
x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
cos = torch.randn(128, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(128, 64, device='cuda', dtype=torch.float16)

output = layer(x, cos, sin)
```

### torch.compile Compatibility

```python
import torch

# Fused ops work with torch.compile
model = MyTransformerLayer(4096, 32, 11008).cuda()
compiled_model = torch.compile(model)

# Use compiled model
output = compiled_model(x, cos, sin)
```

---

## vLLM

### Custom Model with Fused Kernels

```python
# In your vLLM model definition
from vllm.model_executor.layers.linear import LinearMethodBase
from triton_ops import FP8Linear

class FusedvLLMModel:
    """Example vLLM model with fused kernels."""
    
    def __init__(self, config):
        # Use FP8 linear layers for quantization
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

### Serving Configuration

```python
# vLLM serving with optimized kernels
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-2-7b",
    dtype="float16",
    # Enable optimized attention (if supported)
    attention_impl="flash_attn",
)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(["Hello, world!"], sampling_params)
```

---

## Custom Training Loops

### Mixed Precision Training

```python
import torch
from torch.cuda.amp import autocast, GradScaler
from triton_ops import fused_gated_mlp, fp8_gemm

# Setup
model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters())
scaler = GradScaler()

# Training loop
for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast(dtype=torch.float16):
        # Fused ops automatically work with autocast
        output = model(batch)
        loss = compute_loss(output)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, intermediate_dim):
        super().__init__()
        self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim)
    
    def forward(self, x):
        # Use checkpointing to save memory
        return checkpoint(self.mlp, x)
```

---

<div align="center">

**[⬆ Back to Top](#integration-guide)** | **[← Back to Guides](../)**

</div>
