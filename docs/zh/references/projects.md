---
title: 项目
description: "与 Triton Fused Ops 相关或用于对比的开源项目"
---

# 项目

本仓库基于、学习并与以下开源项目进行对比。

## 基础层

| 项目 | 链接 | 与本仓库的关系 |
|:--|:--|:--|
| **OpenAI Triton** | [github.com/triton-lang/triton](https://github.com/triton-lang/triton) | 本仓库所有算子所基于的编译器和 Python DSL。没有 Triton，这些 kernel 将需要用 CUDA C++ 或 CUTLASS C++ 编写。 |
| **PyTorch** | [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch) | 张量运行时和自动微分引擎。我们的算子接受 `torch.Tensor` 输入并返回 `torch.Tensor`，保持 PyTorch 生态契约。 |

## NVIDIA 生态

| 项目 | 链接 | 与本仓库的关系 |
|:--|:--|:--|
| **NVIDIA CUTLASS** | [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | 我们的 FP8 GEMM 设计从 CUTLASS 的混合精度 GEMM 管线中汲取了结构灵感，特别是 grouped GEMM 和 output-tile 排序模式。 |
| **TensorRT-LLM** | [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | 生产级 LLM 推理中融合算子性能的基准线。我们追求在同等硬件上达到可比或更优的延迟。 |
| **FasterTransformer** | [github.com/NVIDIA/FasterTransformer](https://github.com/NVIDIA/FasterTransformer) | 融合模式的历史参考（如 LayerNorm + 激活融合），为我们的算子设计决策提供依据。 |

## 推理框架

| 项目 | 链接 | 与本仓库的关系 |
|:--|:--|:--|
| **vLLM** | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) | 生产推理的集成目标。我们的算子设计为可嵌入 vLLM 风格 serving 系统的原语组件。 |
| **xFormers** | [github.com/facebookresearch/xformers](https://github.com/facebookresearch/xformers) | 高效的 Transformer 原语库。其内存高效注意力和融合算子提供了设计基准。 |
| **DeepSpeed Inference** | [github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) | 分布式推理上下文中的算子融合与量化基准。 |

## 量化

| 项目 | 链接 | 与本仓库的关系 |
|:--|:--|:--|
| **AutoGPTQ** | [github.com/PanQiWei/AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) | 训练后量化工具，与我们仅权重量化场景下的 FP8 GEMM 算子互补。 |
| **bitsandbytes** | [github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | 8-bit 优化器和 LLM.int8() 实现。我们的 `FP8Linear` 模块以 Triton 原生路径追求相同的显存缩减目标。 |
