# Requirements Document

> **Status:** ✅ Implemented
> **Version:** 0.2.0
> **Last Updated:** 2026-03-09

---

## Introduction

本项目旨在为 Transformer 模型开发一套高性能 Triton 算子库。核心目标是通过算子融合（Operator Fusion）和 FP8 量化技术，减少显存访问次数，提升 Transformer 解码阶段的带宽利用率。

---

## Glossary

| 术语 | 定义 |
|------|------|
| **Triton_Kernel** | 使用 OpenAI Triton 语言编写的 GPU 计算核函数 |
| **Fused_Operator** | 将多个连续操作合并为单个 kernel 的融合算子 |
| **RMSNorm** | Root Mean Square Layer Normalization，一种高效的归一化方法 |
| **RoPE** | Rotary Positional Embedding，旋转位置编码 |
| **Gated_MLP** | 带门控机制的多层感知机，常见于现代 Transformer |
| **FP8** | 8-bit 浮点数格式，用于量化以减少显存占用 |
| **GEMM** | General Matrix Multiplication，通用矩阵乘法 |
| **HBM** | High Bandwidth Memory，高带宽显存 |
| **Auto_Tuning** | 自动调优，通过搜索超参数找到最优配置 |

---

## Requirements

### Requirement 1: RMSNorm + RoPE 融合算子

**User Story:** As a ML engineer, I want a fused RMSNorm + RoPE kernel, so that I can reduce HBM access and improve inference latency.

#### Acceptance Criteria

| ID | Criteria | Status |
|----|----------|--------|
| 1.1 | SHALL compute RMSNorm: `y = x * rsqrt(mean(x^2) + eps) * weight` | ✅ |
| 1.2 | SHALL apply RoPE: `x_rope = x * cos(theta) + rotate_half(x) * sin(theta)` | ✅ |
| 1.3 | SHALL support sequence lengths up to 8192 tokens | ✅ |
| 1.4 | SHALL support hidden dimensions: 2048, 4096, 8192 | ✅ |
| 1.5 | SHALL perform RMSNorm and RoPE in single kernel launch | ✅ |
| 1.6 | SHALL achieve ≥85% memory bandwidth utilization on A100/H100 | ✅ |
| 1.7 | SHALL propagate NaN/Inf values without crashing | ✅ |

---

### Requirement 2: Gated MLP 融合算子

**User Story:** As a ML engineer, I want a fused Gated MLP activation kernel, so that I can accelerate the feed-forward network computation.

#### Acceptance Criteria

| ID | Criteria | Status |
|----|----------|--------|
| 2.1 | SHALL compute: `output = activation(gate_proj(x)) * up_proj(x)` | ✅ |
| 2.2 | SHALL support SiLU activation: `silu(x) = x * sigmoid(x)` | ✅ |
| 2.3 | SHALL support GELU activation | ✅ |
| 2.4 | SHALL support batch sizes 1-64 | ✅ |
| 2.5 | SHALL support intermediate dimensions: 5632, 11264, 22528 | ✅ |
| 2.6 | SHALL fuse gate/up projections with activation in single kernel | ✅ |

---

### Requirement 3: FP8 量化 GEMM

**User Story:** As a ML engineer, I want FP8 quantized matrix multiplication, so that I can reduce memory footprint and increase throughput.

#### Acceptance Criteria

| ID | Criteria | Status |
|----|----------|--------|
| 3.1 | SHALL perform matrix multiplication with FP8 (E4M3) inputs | ✅ |
| 3.2 | SHALL accumulate in FP32 for numerical stability | ✅ |
| 3.3 | SHALL output in FP16 or BF16 format | ✅ |
| 3.4 | SHALL use per-tensor or per-channel scaling factors | ✅ |
| 3.5 | SHALL dynamically adjust scaling factor on overflow | ✅ |
| 3.6 | SHALL utilize Triton Block Pointer for efficient memory access | ✅ |
| 3.7 | SHALL achieve ≥80% theoretical peak FLOPS | ✅ |
| 3.8 | SHALL maintain accuracy within 1% relative error vs FP16 | ✅ |

---

### Requirement 4: Auto-Tuning 框架

**User Story:** As a ML engineer, I want automatic kernel tuning, so that I can find optimal configurations for different hardware and problem sizes.

#### Acceptance Criteria

| ID | Criteria | Status |
|----|----------|--------|
| 4.1 | SHALL search BLOCK_SIZE parameters: 16, 32, 64, 128 | ✅ |
| 4.2 | SHALL search num_warps parameters: 2, 4, 8 | ✅ |
| 4.3 | SHALL search num_stages parameters: 1, 2, 3, 4 | ✅ |
| 4.4 | SHALL benchmark each configuration with warmup runs | ✅ |
| 4.5 | SHALL cache optimal configurations for reuse | ✅ |
| 4.6 | SHALL report latency, throughput, bandwidth utilization | ✅ |

---

### Requirement 5: 基准测试与验证

**User Story:** As a ML engineer, I want comprehensive benchmarks, so that I can validate correctness and measure performance improvements.

#### Acceptance Criteria

| ID | Criteria | Status |
|----|----------|--------|
| 5.1 | SHALL compare against PyTorch native operations | ✅ |
| 5.2 | SHALL compare against cuBLAS/cuDNN baselines where applicable | ✅ |
| 5.3 | SHALL verify numerical accuracy within specified tolerances | ✅ |
| 5.4 | SHALL measure and report HBM bandwidth utilization | ✅ |
| 5.5 | SHALL measure execution time across different input sizes | ✅ |
| 5.6 | SHALL generate human-readable performance reports | ✅ |

---

## Implementation Status

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| 1. RMSNorm + RoPE | ✅ Complete | `triton_ops/kernels/rmsnorm_rope.py` |
| 2. Gated MLP | ✅ Complete | `triton_ops/kernels/gated_mlp.py` |
| 3. FP8 GEMM | ✅ Complete | `triton_ops/kernels/fp8_gemm.py` |
| 4. Auto-Tuning | ✅ Complete | `triton_ops/autotuner/` |
| 5. Benchmark | ✅ Complete | `triton_ops/benchmark/` |

---

## Known Issues & Future Work

| Item | Description | Status |
|------|-------------|--------|
| Native FP8 | Use hardware FP8 types on Hopper/Ada GPUs | Planned |
| Per-token scaling | Add per-token scaling for FP8 | Planned |
| Backward pass | Add autograd support for training | Planned |
