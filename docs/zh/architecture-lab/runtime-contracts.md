---
title: 运行时契约
description: Triton Fused Ops 的验证规则、类型化错误与 launch 期假设
---

# 运行时契约

每个交付中的 Kernel family 都假设它的输入已经跨过一条验证边界。这条边界之所以重要，是因为它让 launcher 代码可以保持可读，而不是把 fast path 与零散 guard 逻辑搅在一起。

## validation 实际在检查什么

`triton_ops.validation` 负责拦下那些否则会变成静默陷阱的问题：

- tensor 必须在正确 device 上；
- 必须使用支持的 dtype；
- shape 必须相互匹配；
- contiguous 内存布局要满足预期；
- epsilon、activation 等标量参数必须合理。

## 当前错误词汇

| 错误类型 | 什么时候出现 |
| :-- | :-- |
| `DeviceError` | tensor 不在 CUDA 上，或与其他输入不在同一 device |
| `ShapeMismatchError` | tensor shape 或推导维度不一致 |
| `UnsupportedDtypeError` | dtype 超出 family 支持集合 |
| `ValueError` | contiguous 检查、activation 校验、正标量检查以及部分声明式契约回退仍使用普通 `ValueError` |

`NumericalOverflowError` 与 `TuningFailedError` 也是导出的库异常，但它们并不是当前 `triton_ops.validation` 辅助函数的常规返回结果。

## 按 family 看契约

### RMSNorm + RoPE

验证器会检查 device、dtype、contiguous、`weight` 形状、`cos` 与 `sin` 是否一致，以及 `hidden_dim`、`head_dim`、可选 head 数之间的关系。

### Gated MLP

验证器会检查双投影权重的形状、支持的 activation 选择，以及 `hidden_dim` 与 `intermediate_dim` 之间的关系。

### FP8 栈

FP8 路径会校验 shape 兼容性、scale 是否可用、存储 dtype 是否受支持，以及输出 dtype 的选择是否合法。

## 这对集成为什么关键

当集成失败时，理想目标是尽早失败，并给出当前验证器能提供的最具体解释。这也是为什么文档不断回到运行时契约：即便部分检查仍表现为普通 `ValueError`，它们依然是上游模型代码需要满足的稳定说明书。
