---
title: 博客与文档
description: "技术博客、教程和官方文档资源"
---

# 博客与文档

## NVIDIA 开发者博客

- **Using FP8 with the NVIDIA H100 Tensor Core GPU** — Hopper 架构上 FP8 格式、量化方案和性能预期的官方指南。
  - [developer.nvidia.com/blog](https://developer.nvidia.com/blog)

- **Optimizing Transformer Inference with NVIDIA TensorRT** — 融合模式、kernel 调度与内存优化策略的背景知识，影响我们的算子设计。

## PyTorch 博客

- **torch.compile: A PyTorch Compiler** — 解释 PyTorch 2.0+ 如何通过 TorchInductor 将 Python 编译为 Triton 算子。理解此流水线有助于定位我们的手写 Triton 算子与自动生成算子的关系。
  - [pytorch.org/blog](https://pytorch.org/blog)

## Triton 文档

- **Triton Python API Tutorial** — `@triton.jit` 装饰器、block pointers 和内存合并访问模式的官方教程。
  - [triton-lang.org](https://triton-lang.org)

- **Triton GPU Programming Guide** — 进阶主题：流水线、双缓冲和 warp-level 原语。

## 社区教程

- **GPU Mode: Triton Kernel Development** — 为常见深度学习算子编写自定义 Triton kernel 的实践教程。
  - [gpu-mode.com](https://gpu-mode.com)

- **CUDA Mode Discord & Lectures** — 社区驱动的 GPU 编程、算子优化和性能分析方法论讲座。

## 学术课程资料

- **CS217: Accelerators for Machine Learning (Stanford)** — GPU 架构、内存层次结构和算子优化原理的课堂笔记。
  - [cs217.stanford.edu](https://cs217.stanford.edu)
