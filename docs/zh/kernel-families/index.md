---
title: 算子族
description: 面向用户的融合算子族，以及比较它们时应使用的框架
---

# 算子族

在这个仓库里，**Kernel family** 是最重要的用户侧单位。每一个 family 都对应一段可部署的故事：它解决什么瓶颈、要求什么契约、如何证明自己是对的，以及应该用什么性能语言来评价。

## 对比矩阵

| Kernel family | 主要 API | 工作负载叙事 | 证据表面 |
| :-- | :-- | :-- | :-- |
| [Fused RMSNorm + RoPE](./rmsnorm-rope) | `fused_rmsnorm_rope`、`FusedRMSNormRoPE` | 在旋转位置编码前避免落地中间规范化结果 | validation + reference 实现 + benchmark 对照 |
| [Fused Gated MLP](./gated-mlp) | `fused_gated_mlp`、`FusedGatedMLP` | 把双投影、activation 与 gating 融合进一次 launch | validation + reference 实现 + 按 activation 区分的测量 |
| [FP8 GEMM](./fp8-stack) | `fp8_gemm`、`FP8Linear` | 用精度格式与显式 scale 管理换取存储与吞吐目标 | 量化辅助工具 + scale 语义 + GEMM benchmark |
| FP8 量化工具 | `quantize_fp8`、`dequantize_fp8` | 让 FP8 路径可检查、可解释，而不是黑箱 | 溢出处理、scale 校验、round-trip 语义 |

## 读这些页面时该问什么

对每个 family，都问四个问题：

1. 它试图移动的主瓶颈是什么？
2. 上游必须先准备好哪些 tensor 与契约？
3. 哪条 reference 路径负责核对语义？
4. 应该用哪些 Benchmarking 与 Performance metrics 来评价它？

如果一页文档答不出这四个问题，它就还没真正完成工作。
