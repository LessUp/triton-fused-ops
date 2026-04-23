# Design Document: Triton Fused Operators Library

> **Status:** ✅ Archived
> **Version:** 0.2.0
> **Created:** 2026-03-09
> **Archived:** 2026-04-23

---

## Overview

本设计文档描述了一套高性能 Triton 算子库的架构和实现细节。该库针对 Transformer 模型的解码阶段进行优化，通过算子融合减少 HBM 访问次数，并通过 FP8 量化提升计算吞吐量。

### 核心设计原则

| 原则 | 说明 |
|------|------|
| **最小化内存访问** | 通过算子融合，将多次 HBM 读写合并为单次 |
| **最大化计算密度** | 利用 Triton 的块级并行和寄存器复用 |
| **灵活的精度支持** | 支持 FP32、FP16、BF16、FP8 多种精度 |
| **自动调优** | 通过参数搜索找到最优的 kernel 配置 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Triton Operators Library                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Fused Kernels  │  │   FP8 Kernels   │  │   Auto-Tuner    │  │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤  │
│  │ rmsnorm_rope    │  │ fp8_gemm        │  │ config_search   │  │
│  │ gated_mlp       │  │ fp8_quantize    │  │ benchmark       │  │
│  │                 │  │ fp8_dequantize  │  │ cache_manager   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Core Utilities                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Memory Utils    │  │ Math Primitives │  │ Validation      │  │
│  │ - block_ptr     │  │ - rsqrt         │  │ - correctness   │  │
│  │ - coalesced_io  │  │ - sigmoid       │  │ - numerical     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Python Interface                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ torch.nn.Module wrappers with autograd support              ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Fused RMSNorm + RoPE Kernel

**Purpose:** 将 RMSNorm 和 RoPE 合并为单个内核，减少 HBM 访问。

**Memory Access Pattern:**
```
Without Fusion (3 HBM accesses):
  HBM → RMSNorm → HBM → RoPE → HBM

With Fusion (1 HBM access):
  HBM → [RMSNorm + RoPE in registers] → HBM
```

**Mathematical Formula:**
```
RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
RoPE:    y_rope = y * cos + rotate_half(y) * sin
```

**Status:** ✅ Implemented in `triton_ops/kernels/rmsnorm_rope.py`

---

### 2. Fused Gated MLP Kernel

**Purpose:** 将门控投影、上投影和激活函数融合为单个内核。

**Mathematical Formula:**
```
output = activation(gate_proj(x)) * up_proj(x)

SiLU: silu(x) = x * sigmoid(x)
GELU: gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
```

**Status:** ✅ Implemented in `triton_ops/kernels/gated_mlp.py`

> **Note:** v0.2.0 修复了激活函数应用顺序的 Bug。

---

### 3. FP8 GEMM Kernel

**Purpose:** 使用 FP8 格式进行矩阵乘法，减少显存占用并提升吞吐量。

**FP8 E4M3 Format:**
| Property | Value |
|----------|-------|
| Sign bits | 1 |
| Exponent bits | 4 |
| Mantissa bits | 3 |
| Max value | 448.0 |
| Min normal | 2⁻⁶ |

**Features:**
- FP8 E4M3 输入格式
- FP32 累加器保证数值稳定性
- FP16/BF16 输出
- 动态缩放因子

**Status:** ✅ Implemented in `triton_ops/kernels/fp8_gemm.py`

---

### 4. Auto-Tuning Framework

**Purpose:** 自动搜索最优内核配置。

**Configuration Spaces:**

```python
RMSNORM_ROPE_CONFIGS = {
    'BLOCK_SIZE': [64, 128, 256, 512, 1024],
    'num_warps': [2, 4, 8],
    'num_stages': [1, 2, 3],
}

GATED_MLP_CONFIGS = {
    'BLOCK_M': [32, 64, 128],
    'BLOCK_N': [32, 64, 128],
    'BLOCK_K': [32, 64],
    'num_warps': [4, 8],
    'num_stages': [2, 3, 4],
}

FP8_GEMM_CONFIGS = {
    'BLOCK_M': [64, 128, 256],
    'BLOCK_N': [64, 128, 256],
    'BLOCK_K': [32, 64],
    'GROUP_SIZE_M': [4, 8],
    'num_warps': [4, 8],
    'num_stages': [3, 4, 5],
}
```

**Status:** ✅ Implemented in `triton_ops/autotuner/`

---

## Data Models

### Core Classes

| Class | Purpose | Location |
|-------|---------|----------|
| `TensorSpec` | Tensor specification for validation | `models.py` |
| `RMSNormRoPEInput` | Input spec for RMSNorm + RoPE | `models.py` |
| `GatedMLPInput` | Input spec for Gated MLP | `models.py` |
| `FP8GEMMInput` | Input spec for FP8 GEMM | `models.py` |
| `KernelMetrics` | Performance metrics | `models.py` |
| `TuningResult` | Auto-tuning result | `models.py` |
| `FP8Format` | FP8 format specification | `models.py` |

### Exception Hierarchy

```
TritonKernelError (base)
├── ShapeMismatchError
├── UnsupportedDtypeError
├── NumericalOverflowError
├── TuningFailedError
└── DeviceError
```

---

## Error Handling

### Input Validation

所有内核函数在执行前进行输入验证：

1. **Shape validation** — 确保张量形状兼容
2. **Dtype validation** — 确保数据类型受支持
3. **Device validation** — 确保张量在 CUDA 设备上

### Numerical Safety

- FP8 量化支持动态溢出处理
- NaN/Inf 值正确传播
- 数值容差检查

---

## Correctness Properties

### Property 1: RMSNorm + RoPE Mathematical Correctness

融合内核输出应与顺序应用 RMSNorm + RoPE 数值等价（在浮点容差范围内）。

**Validates:** Requirements 1.1, 1.2

### Property 2: Gated MLP Correctness

融合 Gated MLP 输出应与 `gate_proj(x) * activation(up_proj(x))` 数值等价。

**Validates:** Requirements 2.1, 2.2, 2.3

### Property 3: FP8 GEMM Accuracy

FP8 GEMM 结果与 FP16 基线相比，相对误差应在 1% 以内。

**Validates:** Requirements 3.1, 3.8

### Property 4: FP8 Round-Trip

FP8 量化+反量化往返误差应在预期范围内。

**Validates:** Requirements 3.4

---

## Testing Strategy

### Test Categories

| Category | Purpose | Framework |
|----------|---------|-----------|
| **Property-Based Tests** | 验证通用正确性属性 | Hypothesis |
| **Unit Tests** | 验证特定示例和边界情况 | pytest |
| **Edge Case Tests** | 验证 NaN/Inf、空张量等 | pytest |
| **Benchmark Tests** | 性能测量 | Custom |

### Test Files

```
tests/
├── test_rmsnorm_rope.py      # Property tests for fused RMSNorm + RoPE
├── test_gated_mlp.py         # Property tests for fused Gated MLP
├── test_fp8_gemm.py          # Property tests for FP8 GEMM
├── test_fp8_quantize.py      # Property tests for FP8 quantization
├── test_autotuner.py         # Property tests for auto-tuner
├── test_benchmark.py         # Property tests for benchmark suite
├── test_edge_cases.py        # Unit tests for edge cases
└── benchmarks/
    ├── bench_rmsnorm_rope.py
    ├── bench_gated_mlp.py
    └── bench_fp8_gemm.py
```

### Numerical Tolerances

| Operation | Relative Tolerance | Absolute Tolerance |
|-----------|-------------------|-------------------|
| RMSNorm (FP16) | 1e-3 | 1e-5 |
| RoPE (FP16) | 1e-3 | 1e-5 |
| Gated MLP (FP16) | 1e-3 | 1e-5 |
| FP8 GEMM vs FP16 | 1e-2 | 1e-4 |
| FP8 Round-trip | 1e-2 | 1e-3 |

---

## References

- [OpenAI Triton](https://github.com/openai/triton)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
