---
title: 系统总览
description: 说明公共 API、验证契约、reference 路径与性能工具如何形成一个分层系统
---

# 系统总览

理解 Triton Fused Ops 最干净的方式，是把它看成一个导出面很小、证明面比较完整的分层系统。

## 公共 API surface

`triton_ops.__init__.py` 是仓库对外的主入口。它导出用户可见的 launcher、`nn.Module` 封装、验证契约、数据模型、异常类型、Benchmarking 工具、Auto-Tuning 工具以及 performance 辅助对象。

这样做的价值在于：公共 API 可以保持紧凑，而内部 `*_kernel` Triton 函数继续留在 `triton_ops.kernels`，不进入稳定契约。

## 验证契约

`triton_ops.validation` 负责运行时边界检查。当前验证器会检查：

- device 放置是否正确；
- dtype 是否在支持集合内；
- 内存是否 contiguous；
- shape 是否兼容；
- 标量参数是否合法。

仓库把验证保留为靠近各个 Kernel family 的过程式 helper。这样验证仍然是设计表面的一部分，但不会再引入一套调用者还得额外学习的声明式接口。

## Kernel 与 reference 执行路径

快速路径在 `triton_ops.kernels`，证明路径在 `triton_ops.reference`。

| 路径 | 主要模块 | 存在理由 |
| :-- | :-- | :-- |
| 快速路径 | `triton_ops.kernels.rmsnorm_rope`、`gated_mlp`、`fp8_gemm`、`fp8_quantize` | 在 CUDA 上执行优化后的 Triton kernel |
| 证明路径 | `triton_ops.reference.rmsnorm_rope`、`gated_mlp`、`fp8` | 提供可 CPU 测试、可 GPU 对照的 reference 数学 |

这种 Kernel 与 reference 的分离，是仓库可信度的核心：性能主张旁边始终有一条可直接阅读的语义基线。

## Benchmarking 与 Auto-Tuning

二者相关，但职责不同。

| 关注点 | 主要模块 | 核心问题 |
| :-- | :-- | :-- |
| Benchmarking | `BenchmarkSuite`、`CorrectnessVerifier`、`PerformanceReport` | 某个 Kernel family 在什么方法下测得多快，并且是否保持正确？ |
| Auto-Tuning | `TritonAutoTuner`、`ConfigCache`、配置集合 | 对一个 callable 来说，哪组 launch 配置带来最低 latency？ |
| Performance metrics | `triton_ops.performance` | 给定 latency 与 shape 上下文，能推导出怎样的 throughput 或 bandwidth？ |

这里的区分是刻意保持的：Benchmarking 负责测量与报告，Auto-Tuning 负责配置搜索，Performance metrics 负责派生语言。

## 系统阅读启发式

把仓库按这条链路读下去：

1. 导出的承诺；
2. 契约执行；
3. 快速路径；
4. reference 路径；
5. 测量与解释。

这也是为什么整套文档更像技术白皮书，而不是特性清单。
