---
title: Projects
description: "Open-source projects related to or compared with Triton Fused Ops"
---

# Projects

This repository builds on, learns from, and is compared against the following open-source projects.

## Foundation

| Project | Link | Relation to Triton Fused Ops |
|:--|:--|:--|
| **OpenAI Triton** | [github.com/triton-lang/triton](https://github.com/triton-lang/triton) | The compiler and Python DSL on which every kernel in this repo is built. Without Triton, these kernels would be written in CUDA C++ or CUTLASS C++. |
| **PyTorch** | [github.com/pytorch/pytorch](https://github.com/pytorch/pytorch) | The tensor runtime and autograd engine. Our kernels accept `torch.Tensor` inputs and return `torch.Tensor` outputs, keeping the PyTorch ecosystem contract. |

## NVIDIA Ecosystem

| Project | Link | Relation to Triton Fused Ops |
|:--|:--|:--|
| **NVIDIA CUTLASS** | [github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | Our FP8 GEMM design draws structural inspiration from CUTLASS's mixed-precision GEMM pipelines, especially grouped GEMM and output-tile ordering patterns. |
| **TensorRT-LLM** | [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | A benchmark baseline for fused operator performance in production LLM inference. We aim for comparable or better latency on the same hardware. |
| **FasterTransformer** | [github.com/NVIDIA/FasterTransformer](https://github.com/NVIDIA/FasterTransformer) | Historical reference for fusion patterns (e.g., LayerNorm + activation fusion) that inform our kernel design decisions. |

## Inference Frameworks

| Project | Link | Relation to Triton Fused Ops |
|:--|:--|:--|
| **vLLM** | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) | Integration target for production inference. Our kernels are designed to be drop-in primitives for vLLM-style serving systems. |
| **xFormers** | [github.com/facebookresearch/xformers](https://github.com/facebookresearch/xformers) | A library of efficient Transformers primitives. Their memory-efficient attention and fused operators provide design benchmarks. |
| **DeepSpeed Inference** | [github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) | Kernel fusion and quantization benchmarks in the distributed inference context. |

## Quantization

| Project | Link | Relation to Triton Fused Ops |
|:--|:--|:--|
| **AutoGPTQ** | [github.com/PanQiWei/AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) | Post-training quantization tooling that complements our FP8 GEMM kernel for weight-only quantization workflows. |
| **bitsandbytes** | [github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) | 8-bit optimizers and LLM.int8() implementation. Our `FP8Linear` module targets the same memory-reduction goal with a Triton-native path. |
