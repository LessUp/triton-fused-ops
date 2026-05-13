---
title: Papers
description: "Academic papers behind the techniques used in Triton Fused Ops"
---

# Papers

The techniques in this repository are grounded in published research. Below are the core papers, with a note on how each relates to the codebase.

## Kernel Compilation & Tiling

1. **Tillet, P., Kung, H. T., & Cox, D.** (2019). Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. *WMAS@ASPLOS*.
   - [arXiv:1908.04767](https://arxiv.org/abs/1908.04767)
   - **Relation:** This is the foundational work behind OpenAI Triton, the compiler and IR on which all kernels in this repository are built. The tiling abstraction is the core mechanism for our block-level parallelism.

## Attention & Operator Fusion

2. **Dao, T., et al.** (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS*.
   - [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
   - **Relation:** The fusion philosophy of keeping intermediate values in SRAM rather than writing back to HBM directly inspired our `fused_rmsnorm_rope` and `fused_gated_mlp` design patterns.

3. **Dao, T.** (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.
   - [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
   - **Relation:** Improved work partitioning ideas applicable to kernel tiling strategies.

## FP8 & Quantization

4. **Micikevicius, P., et al.** (2022). FP8 Formats for Deep Learning. *arXiv preprint*.
   - [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)
   - **Relation:** Defines the E4M3/E5M2 formats used as the reference for our `uint8`-based FP8 compatibility representation and scale management.

5. **Dettmers, T., et al.** (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. *NeurIPS*.
   - [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
   - **Relation:** Outlier-aware mixed-precision decomposition informed our overflow-handling helper design.

6. **Xiao, G., et al.** (2023). SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. *ICML*.
   - [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
   - **Relation:** Per-tensor vs per-channel scale migration trade-offs influence our FP8 scale strategy.

7. **Dettmers, T., et al.** (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *arXiv preprint*.
   - [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
   - **Relation:** Low-bit inference ecosystem context; our FP8 path targets similar memory-constrained deployment scenarios.

## Performance Analysis

8. **Williams, S., Waterman, A., & Patterson, D.** (2009). Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures. *Communications of the ACM*.
   - **Relation:** The bandwidth-vs-compute characterization framework we use when classifying kernel bottlenecks (memory-bound vs compute-bound).

9. **Hong, S., & Kim, H.** (2009). An Analytical Model for a GPU Architecture with Memory-Level and Thread-Level Parallelism Awareness. *ISCA*.
   - **Relation:** GPU memory hierarchy modeling that informs our fusion decisions.

## Transformer Architecture

10. **Su, J., et al.** (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv preprint*.
    - [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
    - **Relation:** The RoPE formulation implemented in our `fused_rmsnorm_rope` kernel.

11. **Zhang, B., & Sennrich, R.** (2019). Root Mean Square Layer Normalization. *NeurIPS*.
    - [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
    - **Relation:** RMSNorm formulation used in the fused normalization path.
