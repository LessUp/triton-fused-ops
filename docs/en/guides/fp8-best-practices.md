---
title: FP8 Best Practices
description: "Practical guidance for using the repository's FP8 path safely"
---

# FP8 Best Practices

This repository's FP8 path is useful, but it is not a universal replacement for higher-precision execution. It follows the FP8 format specification described by Micikevicius et al. [1] and draws on quantization strategies from SmoothQuant [2].

## Where FP8 fits well

- matrix-multiplication-heavy inference paths,
- projection layers where moderate quantization error is acceptable,
- memory-sensitive inference workloads.

## Where to stay cautious

- normalization steps,
- numerically fragile output heads,
- any path where you have not compared against an FP16 or BF16 baseline.

## Prefer explicit validation

When you introduce FP8 into a new workload, compare against a higher-precision baseline:

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

## Automatic vs explicit quantization

### Automatic quantization

Use:

```python
out = fp8_gemm(a, b)
```

This is the shortest path and is usually a good starting point.

### Explicit quantization

Use:

```python
from triton_ops import quantize_fp8, fp8_gemm

a_fp8, a_scale = quantize_fp8(a)
b_fp8, b_scale = quantize_fp8(b)
out = fp8_gemm(a_fp8, b_fp8, a_scale, b_scale)
```

This is useful when:

- you want to reuse quantized tensors,
- you want visibility into scale values,
- you want to control when quantization happens.

## Overflow handling

The overflow helper is not part of the root-package export list.

```python
from triton_ops.kernels.fp8_quantize import quantize_fp8_with_overflow_handling
```

Use it when you expect extreme ranges and want retry-based scale reduction before failing with `NumericalOverflowError`.

## Integration advice for `FP8Linear`

`FP8Linear` caches quantized weights after the first forward. That is a strong fit for inference-oriented code, but it means you should be careful about using it in training loops where weights continue to update.

## Rule of thumb

- Keep the numerically sensitive boundaries in higher precision.
- Use FP8 where the memory and throughput trade-off is actually paying off.
- Always measure the model- or workload-level impact, not only the isolated kernel result.

## References

1. Micikevicius, P., et al. (2022). FP8 Formats for Deep Learning. *arXiv preprint*. [arXiv:2209.05433](https://arxiv.org/abs/2209.05433)
2. Xiao, G., et al. (2023). SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. *ICML*. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)

See the [NVIDIA FP8 developer blog](/en/references/blogs) and the full [Papers](/en/references/papers) page for more resources.
