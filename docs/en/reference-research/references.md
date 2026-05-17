---
title: References
description: Curated papers and technical references behind Triton Fused Ops
---

# References

The references below are chosen because they explain either the mathematical content of a Kernel family or the systems language used to evaluate it.

## Core references

| Reference | Why it matters to this repository |
| :-- | :-- |
| FlashAttention | Gives the IO-aware framing that makes fusion discussions concrete instead of hand-wavy. |
| FP8 Formats for Deep Learning | Grounds the FP8 stack in a real format discussion rather than a generic 8-bit narrative. |
| RoFormer | Explains the rotary embedding math used by the RMSNorm + RoPE family. |
| Root Mean Square Layer Normalization | Defines the normalization variant that the fused attention-preparation family uses. |

## Expanded reading list

- **Tillet, Kung, Cox — Triton**: the compiler model that makes these kernels expressible in Python.
- **Dao et al. — FlashAttention / FlashAttention-2**: IO-awareness and work partitioning lessons for fusion-heavy GPU code.
- **Micikevicius et al. — FP8 Formats for Deep Learning**: format semantics for E4M3 and related FP8 reasoning.
- **Dettmers et al. — LLM.int8()**: useful context for error handling and low-precision deployment trade-offs.
- **Xiao et al. — SmoothQuant**: relevant when thinking about scale movement and quantization ergonomics.
- **Su et al. — RoFormer**: rotary embedding formulation.
- **Zhang and Sennrich — Root Mean Square Layer Normalization**: RMSNorm formulation.
- **Williams, Waterman, Patterson — Roofline**: a compact way to interpret bandwidth and throughput trade-offs when discussing Performance metrics.

## Reading heuristic

Use papers to understand *why* a family exists, and use the repository code to understand *how* that idea is operationalized here.
