# Triton Fused Operators Library - Specification

> **Version:** 1.0.1
> **Last Updated:** 2026-04-27

---

## Overview

Triton Fused Operators Library 是一套高性能 Triton 算子库，针对 Transformer 模型的解码阶段进行优化。通过算子融合减少 HBM 访问次数，通过 FP8 量化提升计算吞吐量。

---

## Capabilities

本库提供以下核心能力：

### 1. Fused RMSNorm + RoPE

将 RMSNorm 和 RoPE 合并为单个 kernel，将 HBM 访问从 3 次减少到 1 次。

**Supports:**
- 序列长度：1-8192 tokens
- 隐藏维度：2048, 4096, 8192
- 带宽利用率：≥85% on A100/H100

### 2. Fused Gated MLP

融合门控投影、上投影和激活函数。

**Supports:**
- SiLU 和 GELU 激活函数
- 批次大小：1-64
- 中间维度：5632, 11264, 22528

### 3. FP8 GEMM

使用 FP8 E4M3 格式进行矩阵乘法。

**Features:**
- FP8 输入，FP16/BF16 输出
- FP32 累加器保证数值稳定性
- 动态缩放因子
- 理论峰值 FLOPS：≥80%

### 4. Auto-Tuning

自动搜索最优 kernel 配置。

**Features:**
- 可配置的参数空间
- 配置缓存机制
- 性能指标报告

---

## History

| Date | Change | Description |
|------|--------|-------------|
| 2026-03-09 | triton-fused-operators | Initial implementation of fused operators |
| 2026-04-16 | project-standardization | Project documentation and CI/CD setup |

---

## References

- [OpenAI Triton](https://github.com/openai/triton)
- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
