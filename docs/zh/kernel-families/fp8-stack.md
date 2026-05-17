---
title: FP8 栈
description: FP8 GEMM、FP8 量化工具与 scale 管理策略
---

# FP8 栈

仓库里的 FP8 路径不是单一 kernel，而是一整套 stack：包括 FP8 量化工具、反量化、`FP8 GEMM`，以及为重复推理缓存量化权重的 `FP8Linear` 封装。

## 栈内组件

| 组件 | 角色 |
| :-- | :-- |
| `quantize_fp8` | 把浮点 tensor 转成仓库兼容的 FP8 `uint8` 存储格式 |
| `dequantize_fp8` | 结合存储值与 scale 恢复 FP16 / BF16 tensor |
| `fp8_gemm` | 带显式 scale、并在 FP32 中累积的矩阵乘 |
| `FP8Linear` | 用熟悉的线性层接口包装 FP8 GEMM |

## 格式与存储模型

实现对两点保持显式：

1. 逻辑格式以 E4M3 风格 FP8 为中心，常量由 `FP8Format` 锚定；
2. 实际存储路径使用 `uint8` 加 `scale` tensor，因此即便环境对原生 FP8 tensor 的支持不一致，工作流依然可移植。

正因为格式、溢出处理与转换路径都写在代码里，这个 family 才是可审查、evidence-backed 的。

## scale 管理

`quantize_fp8` 可以接收外部给定的 `scale`，也可以通过 `FP8Format.compute_scale` 自动计算。`dequantize_fp8` 与 `fp8_gemm` 依赖同一套 scale 语义；如果 `scale` 无效，代码会抛出数值或验证错误，而不是自行猜测。

## FP8 GEMM 的运行时故事

`fp8_gemm` 会在需要时自动量化浮点输入，校验 shape 与 dtype 兼容性，然后启动一个在 FP32 中累积、最后再转成目标输出 dtype 的 Triton GEMM。

kernel 还使用 grouped tile ordering 改善 cache 行为，这是叠加在格式与 scale 故事之上的系统级优化。

## 什么时候值得 benchmark

只有当你的部署问题真正落在权重占用与矩阵乘吞吐上时，FP8 Benchmarking 才最有意义；若主成本在模型其他路径，就不应过度解读。只有矩阵形状明确后，Performance metrics 才适合加入讨论。
