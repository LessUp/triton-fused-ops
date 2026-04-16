---
layout: default
title: "Performance Tuning — Triton Fused Ops"
description: "Performance tuning guide for Triton Fused Ops - GPU optimization strategies"
---

# Performance Tuning Guide

Optimize Triton Fused Ops for your specific hardware.

---

## 📑 Table of Contents

- [GPU-Specific Optimization](#gpu-specific-optimization)
- [Memory Bandwidth](#memory-bandwidth)
- [Auto-Tuning](#auto-tuning)
- [Batch Size Optimization](#batch-size-optimization)
- [Profiling](#profiling)

---

## GPU-Specific Optimization

### Recommended Configurations by GPU

| GPU | BLOCK_M | BLOCK_N | num_warps | Notes |
|:----|:--------|:--------|:----------|:------|
| **A100 80GB** | 128 | 128 | 8 | Balanced for large matrices |
| **A100 40GB** | 128 | 64 | 8 | Memory-constrained configs |
| **H100** | 256 | 128 | 8 | Larger blocks for H100 |
| **RTX 4090** | 128 | 128 | 4 | Fewer warps for Ada |
| **A6000** | 128 | 128 | 8 | Similar to A100 |

### Environment Variables

```bash
# CUDA optimization
export CUDA_LAUNCH_BLOCKING=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Triton cache
export TRITON_CACHE_DIR=~/.triton/cache

# PyTorch CUDA settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## Memory Bandwidth

### Understanding Memory Bound vs Compute Bound

```
Memory Bandwidth Bound:
┌─────────────────────────────────────────┐
│  Fused RMSNorm+RoPE: 90%+ bandwidth     │
│  Small matrices, element-wise ops       │
│  Solution: Maximize parallelism         │
└─────────────────────────────────────────┘

Compute Bound:
┌─────────────────────────────────────────┐
│  FP8 GEMM: 60-70% bandwidth             │
│  Large matrices, compute-heavy          │
│  Solution: Optimize block sizes         │
└─────────────────────────────────────────┘
```

### Measuring Bandwidth Utilization

```python
import torch
from triton_ops import fused_rmsnorm_rope

def measure_bandwidth(kernel_fn, *args, bytes_per_element=2):
    """Measure effective memory bandwidth."""
    
    # Calculate total bytes read/written
    total_bytes = sum(
        arg.numel() * bytes_per_element 
        for arg in args if isinstance(arg, torch.Tensor)
    )
    
    # Time the kernel
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(100):
        output = kernel_fn(*args)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start.elapsed_time(end) / 100
    bandwidth_gbps = (total_bytes / (1024**3)) / (elapsed_ms / 1000)
    
    return bandwidth_gbps

# Measure
x = torch.randn(8, 2048, 4096, device='cuda', dtype=torch.float16)
weight = torch.ones(4096, device='cuda', dtype=torch.float16)
cos = torch.randn(2048, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(2048, 64, device='cuda', dtype=torch.float16)

bandwidth = measure_bandwidth(
    fused_rmsnorm_rope, x, weight, cos, sin
)
print(f"Effective bandwidth: {bandwidth:.1f} GB/s")

# A100 peak: ~2000 GB/s
# Good utilization: >80%
```

---

## Auto-Tuning

### Finding Optimal Configurations

```python
from triton_ops import TritonAutoTuner, FP8_GEMM_CONFIGS
import torch

def my_gemm(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8):
    # Your kernel implementation
    pass

# Create tuner
tuner = TritonAutoTuner(
    kernel_fn=my_gemm,
    config_space={
        "BLOCK_M": [64, 128, 256],
        "BLOCK_N": [64, 128, 256],
        "BLOCK_K": [32, 64],
        "num_warps": [4, 8],
    },
    warmup_runs=10,
    benchmark_runs=50,
)

# Tune for your problem size
M, N, K = 4096, 4096, 4096
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

result = tuner.tune(
    a, b,
    problem_size=(M, N, K),
    device=torch.cuda.get_device_name(),
)

print(f"Optimal config: {result.best_config}")
print(f"Latency: {result.metrics.latency_ms:.3f} ms")
```

### Caching Results

```python
from triton_ops import ConfigCache

# Persistent cache
cache = ConfigCache(cache_dir="~/.triton_config_cache")

# Store optimal config
cache.set(
    kernel_type="fp8_gemm",
    problem_size=(4096, 4096, 4096),
    device="NVIDIA A100-SXM4-80GB",
    config=result.best_config,
)

# Retrieve later
cached_config = cache.get(
    kernel_type="fp8_gemm",
    problem_size=(4096, 4096, 4096),
    device="NVIDIA A100-SXM4-80GB",
)
```

---

## Batch Size Optimization

### Optimal Batch Sizes

| Operation | Small Batch (1-4) | Medium (8-16) | Large (32+) |
|:----------|:------------------|:--------------|:------------|
| **RMSNorm+RoPE** | Good | Excellent | Excellent |
| **Gated MLP** | Good | Excellent | Excellent |
| **FP8 GEMM** | Fair | Good | Excellent |

### Dynamic Batch Size Handling

```python
import torch
from triton_ops import fused_rmsnorm_rope

class DynamicBatcher:
    """Handle variable batch sizes efficiently."""
    
    def __init__(self, max_batch_size=64):
        self.max_batch_size = max_batch_size
        self.cache = {}
    
    def get_optimal_config(self, batch_size):
        """Get pre-tuned config for batch size."""
        if batch_size not in self.cache:
            # Run auto-tuning for this batch size
            self.cache[batch_size] = self._tune_for_batch(batch_size)
        return self.cache[batch_size]
    
    def _tune_for_batch(self, batch_size):
        # Implement tuning logic
        pass
```

---

## Profiling

### Using PyTorch Profiler

```python
import torch
from torch.profiler import profile, ProfilerActivity
from triton_ops import fused_rmsnorm_rope

# Prepare inputs
x = torch.randn(8, 2048, 4096, device='cuda', dtype=torch.float16)
weight = torch.ones(4096, device='cuda', dtype=torch.float16)
cos = torch.randn(2048, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(2048, 64, device='cuda', dtype=torch.float16)

# Profile
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    for _ in range(10):
        output = fused_rmsnorm_rope(x, weight, cos, sin)

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export for visualization
prof.export_chrome_trace("trace.json")
```

### Nsight Systems

```bash
# Profile with Nsight Systems
nsys profile -o profile_report \
    python your_script.py

# View results
nsys-ui profile_report.nsys-rep
```

### Nsight Compute

```bash
# Detailed kernel analysis
ncu --kernel-name my_kernel \
    --metrics dram__bytes_read.sum,dram__bytes_write.sum \
    python your_script.py
```

---

<div align="center">

**[⬆ Back to Top](#performance-tuning-guide)** | **[← Back to Guides](../)**

</div>
