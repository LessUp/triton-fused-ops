# Requirements Document

## Introduction

本项目旨在为 gpt-oss-20b（假设的 OpenAI 开源模型）开发一套高性能 Triton 算子库。核心目标是通过算子融合（Operator Fusion）和 FP8 量化技术，减少显存访问次数，提升 Transformer 解码阶段的带宽利用率。项目展示"不依赖 NVIDIA 闭源库"的异构计算能力。

## Glossary

- **Triton_Kernel**: 使用 OpenAI Triton 语言编写的 GPU 计算核函数
- **Fused_Operator**: 将多个连续操作合并为单个 kernel 的融合算子，减少中间结果的显存读写
- **RMSNorm**: Root Mean Square Layer Normalization，一种高效的归一化方法
- **RoPE**: Rotary Positional Embedding，旋转位置编码，用于注入位置信息
- **Gated_MLP**: 带门控机制的多层感知机，常见于现代 Transformer 架构
- **FP8**: 8-bit 浮点数格式，用于量化以减少显存占用和提升计算吞吐
- **GEMM**: General Matrix Multiplication，通用矩阵乘法
- **HBM**: High Bandwidth Memory，高带宽显存
- **Block_Pointer**: Triton 的块级指针特性，用于高效的内存访问模式
- **Auto_Tuning**: 自动调优，通过搜索超参数（如 BLOCK_SIZE、num_warps）找到最优配置
- **Scaling_Factor**: 量化过程中用于缩放数值范围的因子

## Requirements

### Requirement 1: RMSNorm + RoPE 融合算子

**User Story:** As a ML engineer, I want a fused RMSNorm + RoPE kernel, so that I can reduce HBM access and improve inference latency.

#### Acceptance Criteria

1. THE Fused_Operator SHALL compute RMSNorm according to the formula: `y = x * rsqrt(mean(x^2) + eps) * weight`
2. THE Fused_Operator SHALL apply RoPE transformation using the formula: `x_rope = x * cos(theta) + rotate_half(x) * sin(theta)`
3. WHEN processing a sequence, THE Fused_Operator SHALL support variable sequence lengths up to 8192 tokens
4. THE Fused_Operator SHALL support hidden dimensions of 2048, 4096, and 8192
5. WHEN the input tensor is provided, THE Fused_Operator SHALL perform RMSNorm and RoPE in a single kernel launch without intermediate HBM writes
6. THE Fused_Operator SHALL achieve at least 85% memory bandwidth utilization on NVIDIA A100/H100 GPUs
7. IF the input contains NaN or Inf values, THEN THE Fused_Operator SHALL propagate these values without crashing

### Requirement 2: Gated MLP 融合算子

**User Story:** As a ML engineer, I want a fused Gated MLP activation kernel, so that I can accelerate the feed-forward network computation.

#### Acceptance Criteria

1. THE Fused_Operator SHALL compute the gated activation: `output = gate_proj(x) * activation(up_proj(x))`
2. THE Fused_Operator SHALL support SiLU (Swish) activation function: `silu(x) = x * sigmoid(x)`
3. THE Fused_Operator SHALL support GELU activation function as an alternative
4. WHEN processing batched inputs, THE Fused_Operator SHALL handle batch sizes from 1 to 64
5. THE Fused_Operator SHALL support intermediate dimensions of 5632, 11264, and 22528 (typical for 7B, 13B, 20B models)
6. WHEN gate_proj and up_proj weights are provided, THE Fused_Operator SHALL fuse the two projections with activation in a single kernel

### Requirement 3: FP8 量化 GEMM

**User Story:** As a ML engineer, I want FP8 quantized matrix multiplication, so that I can reduce memory footprint and increase throughput.

#### Acceptance Criteria

1. THE Triton_Kernel SHALL perform matrix multiplication with FP8 (E4M3 format) inputs
2. THE Triton_Kernel SHALL accumulate results in FP32 to maintain numerical stability
3. THE Triton_Kernel SHALL output results in FP16 or BF16 format
4. WHEN quantizing from FP16 to FP8, THE Triton_Kernel SHALL use per-tensor or per-channel scaling factors
5. IF overflow is detected during FP8 conversion, THEN THE Triton_Kernel SHALL dynamically adjust the scaling factor
6. THE Triton_Kernel SHALL utilize Triton Block_Pointer for efficient memory access patterns
7. THE Triton_Kernel SHALL achieve at least 80% of theoretical peak FLOPS on supported hardware
8. WHEN compared to FP16 baseline, THE Triton_Kernel SHALL maintain accuracy within 1% relative error for typical model weights

### Requirement 4: Auto-Tuning 框架

**User Story:** As a ML engineer, I want automatic kernel tuning, so that I can find optimal configurations for different hardware and problem sizes.

#### Acceptance Criteria

1. THE Auto_Tuning framework SHALL search over BLOCK_SIZE parameters (16, 32, 64, 128)
2. THE Auto_Tuning framework SHALL search over num_warps parameters (2, 4, 8)
3. THE Auto_Tuning framework SHALL search over num_stages parameters (1, 2, 3, 4)
4. WHEN tuning is requested, THE Auto_Tuning framework SHALL benchmark each configuration with warmup runs
5. THE Auto_Tuning framework SHALL cache optimal configurations for reuse
6. THE Auto_Tuning framework SHALL report performance metrics including latency, throughput, and bandwidth utilization

### Requirement 5: 基准测试与验证

**User Story:** As a ML engineer, I want comprehensive benchmarks, so that I can validate correctness and measure performance improvements.

#### Acceptance Criteria

1. THE Benchmark_Suite SHALL compare Triton implementations against PyTorch native operations
2. THE Benchmark_Suite SHALL compare Triton implementations against cuBLAS/cuDNN baselines where applicable
3. WHEN running correctness tests, THE Benchmark_Suite SHALL verify numerical accuracy within specified tolerances
4. THE Benchmark_Suite SHALL measure and report HBM bandwidth utilization
5. THE Benchmark_Suite SHALL measure and report kernel execution time across different input sizes
6. THE Benchmark_Suite SHALL generate performance reports in a human-readable format
