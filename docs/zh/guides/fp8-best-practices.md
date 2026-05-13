---
title: FP8 最佳实践
description: "如何更稳妥地使用仓库中的 FP8 路径"
---

# FP8 最佳实践

本仓库的 FP8 路径很有价值，但它不是所有场景下都可以直接替代高精度计算。它遵循 Micikevicius 等人 [1] 描述的 FP8 格式规范，并借鉴了 SmoothQuant [2] 的量化策略。

## 哪些位置更适合 FP8

- 以矩阵乘为主的推理路径，
- 对投影层的少量量化误差相对可接受的场景，
- 显存与带宽压力比较突出的推理负载。

## 哪些位置要更谨慎

- 归一化步骤，
- 数值敏感的输出头，
- 任何尚未和 FP16/BF16 baseline 对齐验证过的路径。

## 优先做显式验证

引入 FP8 之前，先和更高精度 baseline 对比：

```python
import torch
from triton_ops import fp8_gemm

a = torch.randn(256, 512, device="cuda", dtype=torch.float16) * 0.02
b = torch.randn(512, 256, device="cuda", dtype=torch.float16) * 0.02

fp16_out = torch.matmul(a, b)
fp8_out = fp8_gemm(a, b)

rel_error = (fp8_out.float() - fp16_out.float()).abs() / (fp16_out.float().abs() + 1e-6)
print(rel_error.mean().item())
```

## 自动量化 vs 显式量化

### 自动量化

```python
out = fp8_gemm(a, b)
```

最短路径，通常适合作为起点。

### 显式量化

```python
from triton_ops import quantize_fp8, fp8_gemm

a_fp8, a_scale = quantize_fp8(a)
b_fp8, b_scale = quantize_fp8(b)
out = fp8_gemm(a_fp8, b_fp8, a_scale, b_scale)
```

适合场景：

- 你要复用量化后的 tensor，
- 你想观察 scale 的变化，
- 你想控制量化发生的具体时机。

## 溢出处理

带溢出处理的 helper 不在根包导出列表中：

```python
from triton_ops.kernels.fp8_quantize import quantize_fp8_with_overflow_handling
```

当你怀疑输入范围过大时，可以用它在失败前先自动缩小 scale；若仍无法解决，则会抛出 `NumericalOverflowError`。

## `FP8Linear` 的使用建议

`FP8Linear` 会在首次前向后缓存量化权重，这更适合推理型代码；如果你处在持续更新权重的训练循环中，需要特别注意这个缓存语义。

## 经验法则

- 数值敏感边界继续保留高精度。
- 只有在吞吐和显存收益真实存在的地方再上 FP8。
- 一定要测 workload 级或模型级收益，而不是只看单个 kernel 的局部提速。

## 参考文献

1. Micikevicius, P., et al. (2022). FP8 Formats for Deep Learning. *arXiv preprint*. [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)
2. Xiao, G., et al. (2023). SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. *ICML*. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)

详见 [NVIDIA FP8 开发者博客](/zh/references/blogs) 与完整 [论文](/zh/references/papers) 页面获取更多资源。
