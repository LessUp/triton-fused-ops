---
title: System Overview
description: The system-level explanation of public APIs, validation, reference execution, and tooling
---

# System Overview

The cleanest way to understand Triton Fused Ops is as a layered system with a small exported surface and several supporting proof surfaces behind it.

## Public API surface

`triton_ops.__init__.py` is the main public face of the repository. It exports the user-facing launchers, `nn.Module` wrappers, validation contracts, data models, exception types, Benchmarking helpers, Auto-Tuning helpers, and performance utilities.

That choice matters because it keeps the public story compact: users import kernel families and support tools from one place, while internal `*_kernel` Triton functions stay private to `triton_ops.kernels`.

## Validation contracts

`triton_ops.validation` owns the runtime boundary checks. The validators inspect:

- device placement,
- supported dtypes,
- contiguity,
- shape compatibility,
- scalar parameter validity.

The repo also exposes declarative contract objects such as `RMSNORM_ROPE_CONTRACT`, `GATED_MLP_CONTRACT`, `FP8_GEMM_CONTRACT`, and `FP8_QUANTIZE_CONTRACT`. That means validation is not just ad hoc error handling; it is part of the design surface.

## Kernel and reference execution

The fast path lives in `triton_ops.kernels`. The proof path lives in `triton_ops.reference`.

### Kernel and reference execution

| Path | Main modules | Why it exists |
| :-- | :-- | :-- |
| Fast path | `triton_ops.kernels.rmsnorm_rope`, `gated_mlp`, `fp8_gemm`, `fp8_quantize` | Execute optimized Triton code on CUDA hardware |
| Proof path | `triton_ops.reference.rmsnorm_rope`, `gated_mlp`, `fp8` | Supply CPU-testable or GPU-comparable reference math |

This split is central to the repository’s credibility. The library does not ask readers to trust performance claims in isolation; it keeps a nearby implementation that can be reasoned about more directly.

## Benchmarking and Auto-Tuning

These are related, but they are not the same job.

| Concern | Main module | Core question |
| :-- | :-- | :-- |
| Benchmarking and Auto-Tuning | `triton_ops.benchmark`, `triton_ops.autotuner` | How do we measure or search without mixing the two? |
| Benchmarking | `BenchmarkSuite`, `CorrectnessVerifier`, `PerformanceReport` | How fast was a Kernel family, under what method, and did it stay correct? |
| Auto-Tuning | `TritonAutoTuner`, `ConfigCache`, config sets | Which launch configuration minimizes latency for a callable? |
| Performance metrics | `triton_ops.performance` | Given latency and shape context, what throughput or bandwidth does that imply? |

The distinction from CONTEXT.md is deliberate:

- Benchmarking owns measurement and reporting.
- Auto-Tuning owns configuration search.
- Performance metrics own derived throughput and bandwidth language.

## System reading heuristic

Read the repository as a chain:

1. exported promise,
2. contract enforcement,
3. fast path,
4. reference path,
5. measurement and interpretation.

That order is the reason the docs read like a whitepaper rather than a feature list.
