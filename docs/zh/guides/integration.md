---
title: 集成指南
description: 如何在不违反运行时契约的前提下把 Triton Fused Ops 接入更大的推理代码
---

# 集成指南

这一页讨论的是：如何把某个 **Kernel family** 嵌入现有推理代码，同时保持运行时契约清晰可见。

## 从边界开始，而不是从 kernel 开始

在动模型代码之前，先回答三个问题：

1. 现有 block 已经持有哪些 tensor？
2. 哪些运行时契约已经由上游满足？
3. 这次替换是整块替换，还是只替换一个热点子路径？

这样做能避免常见错误：先选 kernel，再反过来强迫模型代码补齐它并不持有的状态。

## 函数式 API vs 模块封装

| 选项 | 适用场景 | 相关 API |
| :-- | :-- | :-- |
| 函数式调用 | 你已经持有权重、cache 与 tensor 编排逻辑 | `fused_rmsnorm_rope`、`fused_gated_mlp`、`fp8_gemm`、`quantize_fp8`、`dequantize_fp8` |
| 模块封装 | 你希望边界以 `nn.Module` 形式复用 | `FusedRMSNormRoPE`、`FusedGatedMLP`、`FP8Linear` |

## 必须满足的运行时契约

launcher 假设上游代码已经提供 CUDA 上的 tensor、受支持的 dtype、兼容的 shape，以及 contiguous 布局。如果这些前提还没成立，应先修调用方，而不是先怀疑 Kernel family。

## 分 family 的接入提示

### `FusedRMSNormRoPE`

`FusedRMSNormRoPE` 适合放在模型已经持有 RoPE cache，或能够廉价地产生 cache 的边界。它不是一般化 norm 层替换件，因为契约里显式包含 `cos` 与 `sin`。

### `FusedGatedMLP`

这个模块只覆盖 gated expansion 阶段，而不是完整 feed-forward block。down projection、residual 路径以及 block 级编排仍应留在模块外层，除非你在上层再包一层更大的抽象。

### `FP8Linear`

`FP8Linear` 是围绕 FP8 栈提供的推理导向封装。只有当权重足够稳定、量化存储与缓存 scale 真正有价值时，它才是合理取舍。

## 集成检查清单

- 先对照 `triton_ops.validation` 核对运行时契约；
- rollout 前用 reference 路径对比输出；
- benchmark 你真正改动的模型边界，而不是只测孤立 kernel；
- 在理解局部替换之前，不要顺手改动模型其余部分。
