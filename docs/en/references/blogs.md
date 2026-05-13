---
title: Blogs & Documentation
description: "Technical blogs, tutorials, and official documentation resources"
---

# Blogs & Documentation

## NVIDIA Developer Blog

- **Using FP8 with the NVIDIA H100 Tensor Core GPU** — Official guide to FP8 formats, quantization recipes, and performance expectations on Hopper architecture.
  - [developer.nvidia.com/blog](https://developer.nvidia.com/blog)

- **Optimizing Transformer Inference with NVIDIA TensorRT** — Background on fusion patterns, kernel scheduling, and memory optimization strategies that inform our kernel design.

## PyTorch Blog

- **torch.compile: A PyTorch Compiler** — Explains how PyTorch 2.0+ compiles Python into Triton kernels via TorchInductor. Understanding this pipeline helps position our hand-written Triton kernels relative to auto-generated ones.
  - [pytorch.org/blog](https://pytorch.org/blog)

## Triton Documentation

- **Triton Python API Tutorial** — Official walkthrough of `@triton.jit` decorators, block pointers, and memory coalescing patterns.
  - [triton-lang.org](https://triton-lang.org)

- **Triton GPU Programming Guide** — Advanced topics: pipelining, double-buffering, and warp-level primitives.

## Community Tutorials

- **GPU Mode: Triton Kernel Development** — Practical tutorials on writing custom Triton kernels for common deep-learning operations.
  - [gpu-mode.com](https://gpu-mode.com)

- **CUDA Mode Discord & Lectures** — Community-driven lectures on GPU programming, kernel optimization, and profiling methodology.

## Academic Course Materials

- **CS217: Accelerators for Machine Learning (Stanford)** — Lecture notes on GPU architecture, memory hierarchies, and kernel optimization principles.
  - [cs217.stanford.edu](https://cs217.stanford.edu)
