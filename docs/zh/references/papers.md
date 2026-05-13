---
title: 论文
description: "Triton Fused Ops 所用技术背后的学术论文"
---

# 论文

本仓库的技术建立在已发表的研究之上。以下是核心论文及其与本代码库的关系。

## 内核编译与分块

1. **Tillet, P., Kung, H. T., & Cox, D.** (2019). Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. *WMAS@ASPLOS*.
   - [arXiv:1908.04767](https://arxiv.org/abs/1908.04767)
   - **关系：** OpenAI Triton 的奠基之作，本仓库所有算子的编译器和 IR 基础。分块抽象是我们块级并行机制的核心。

## 注意力机制与算子融合

2. **Dao, T., et al.** (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS*.
   - [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
   - **关系：** 将中间值保留在 SRAM 而非写回 HBM 的融合思想，直接启发了 `fused_rmsnorm_rope` 和 `fused_gated_mlp` 的设计模式。

3. **Dao, T.** (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.
   - [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
   - **关系：** 更优的工作划分思想，适用于 kernel tiling 策略改进。

## FP8 与量化

4. **Micikevicius, P., et al.** (2022). FP8 Formats for Deep Learning. *arXiv preprint*.
   - [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)
   - **关系：** 定义了 E4M3/E5M2 格式，是我们 `uint8` 兼容 FP8 表示和 scale 管理的参考依据。

5. **Dettmers, T., et al.** (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. *NeurIPS*.
   - [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
   - **关系：** 异常值感知的混合精度分解，影响了我们溢出处理 helper 的设计。

6. **Xiao, G., et al.** (2023). SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. *ICML*.
   - [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
   - **关系：** 逐张量 vs 逐通道 scale 迁移的权衡，影响了我们的 FP8 scale 策略。

7. **Dettmers, T., et al.** (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv preprint*.
   - [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
   - **关系：** 低比特推理生态背景；我们的 FP8 路径面向类似的显存受限部署场景。

## 性能分析

8. **Williams, S., Waterman, A., & Patterson, D.** (2009). Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures. *Communications of the ACM*.
   - **关系：** 带宽与计算量化的表征框架，我们用于分类 kernel 瓶颈（memory-bound vs compute-bound）。

9. **Hong, S., & Kim, H.** (2009). An Analytical Model for a GPU Architecture with Memory-Level and Thread-Level Parallelism Awareness. *ISCA*.
   - **关系：** GPU 内存层次建模，为我们的融合决策提供理论依据。

## Transformer 架构

10. **Su, J., et al.** (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv preprint*.
    - [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
    - **关系：** `fused_rmsnorm_rope` 中实现的 RoPE 公式来源。

11. **Zhang, B., & Sennrich, R.** (2019). Root Mean Square Layer Normalization. *NeurIPS*.
    - [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
    - **关系：** 融合归一化路径中使用的 RMSNorm 公式来源。
