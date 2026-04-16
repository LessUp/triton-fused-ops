---
layout: default
title: "基准测试 API — Triton Fused Ops"
description: "Triton Fused Ops 基准测试工具 API 参考 - 性能测量和正确性验证"
---

# 基准测试 API 参考

本文档提供基准测试套件的 API 参考。

---

## 概述

基准测试套件提供以下工具：
- 带同步的性能测量
- 与 PyTorch 参考的正确性验证
- 报告生成
- 基准测试编排

---

## 基准测试函数

### benchmark_kernel

使用适当的预热和同步对算子函数进行基准测试。

```python
from triton_ops.benchmark.suite import benchmark_kernel

results = benchmark_kernel(
    kernel_fn=my_kernel,
    args=(input_tensor, weight),
    warmup=10,
    iterations=100,
    device='cuda',
)

print(f"平均延迟: {results['mean_ms']:.3f} ms")
print(f"标准差: {results['std_ms']:.3f} ms")
```

### compare_correctness

将算子输出与 PyTorch 参考进行比较。

```python
from triton_ops.benchmark.correctness import compare_correctness

is_correct, max_error = compare_correctness(
    kernel_fn=fused_rmsnorm_rope,
    reference_fn=pytorch_rmsnorm_rope,
    args=(x, weight, cos, sin),
    rtol=1e-3,
    atol=1e-5,
)

if is_correct:
    print(f"✅ 正确! 最大误差: {max_error:.6f}")
else:
    print(f"❌ 不正确! 最大误差: {max_error:.6f}")
```

---

## 基准测试报告

生成格式化的基准测试报告。

```python
from triton_ops.benchmark.report import BenchmarkReport

report = BenchmarkReport()

# 添加结果
report.add_result(
    name="RMSNorm+RoPE",
    config={"batch": 8, "seq_len": 2048},
    metrics={"latency_ms": 0.89, "speedup": 3.2},
)

# 生成报告
print(report.to_markdown())
print(report.to_json())
```

---

## 性能指标

### 延迟测量

```python
import torch
import time

def measure_latency(fn, *args, warmup=10, iterations=100):
    """使用适当同步测量算子延迟。"""
    
    # 预热
    for _ in range(warmup):
        _ = fn(*args)
    torch.cuda.synchronize()
    
    # 基准测试
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    times = []
    for _ in range(iterations):
        start.record()
        output = fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
    }
```

### 内存带宽计算

```python
def compute_bandwidth(tensor_size_bytes, latency_ms):
    """计算内存带宽（GB/s）。"""
    seconds = latency_ms / 1000
    gigabytes = tensor_size_bytes / (1024**3)
    return gigabytes / seconds
```

---

<div align="center">

**[⬆ 返回顶部](#基准测试-api-参考)** | **[← 返回 API 索引](./)**

</div>
