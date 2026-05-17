---
title: Overview
description: A high-signal orientation to Triton Fused Ops, its vocabulary, and its evidence model
---

# Overview

This section is the shortest path to understanding what the repository is, what it is not, and how to read the rest of the docs without mixing terms.

## Project posture

Triton Fused Ops is a kernel library for Transformer inference. The repository is organized around a small set of **Kernel family** entry points, plus support layers for validation, reference execution, Benchmarking, Auto-Tuning, and Performance metrics.

The practical stance is industrial rather than expansive:

- optimize a narrow set of hot-path operations,
- keep correctness auditable through reference implementations,
- keep runtime contracts explicit,
- keep performance claims attached to measurement method.

## Core vocabulary

| Term | Meaning in this repository | What it does **not** mean |
| :-- | :-- | :-- |
| **Kernel family** | A user-facing fused operation such as `fused_rmsnorm_rope`, `fused_gated_mlp`, or `fp8_gemm` | A vague synonym for any internal Triton kernel |
| **Benchmarking** | Repository tooling that verifies correctness, measures latency, and reports results for a Kernel family | Auto-Tuning or generic profiling |
| **Auto-Tuning** | Configuration search over launch parameters with cached lowest-latency results | End-to-end benchmarking or runtime magic |
| **Performance metrics** | Derived throughput and bandwidth figures computed from latency plus shape context | Raw latency alone |

## What ships in the box

| Layer | Main code paths | Reason it exists |
| :-- | :-- | :-- |
| Public API surface | `triton_ops.__init__.py` | Give users one stable import surface |
| Validation layer | `triton_ops.validation` | Reject invalid device, dtype, shape, and contiguity combinations before launch |
| Kernel layer | `triton_ops.kernels.*` | Run the optimized Triton implementation |
| Reference layer | `triton_ops.reference.*` | Provide CPU-testable and correctness-checkable baselines |
| Tooling layer | `triton_ops.benchmark`, `triton_ops.autotuner`, `triton_ops.performance` | Measure, compare, and interpret kernel behavior |

## Evidence model

A technical whitepaper is only useful if readers can trace each claim back to a mechanism.

1. **Capability claims** should be visible in the exported API.
2. **Correctness claims** should be tied to `triton_ops.reference` and `CorrectnessVerifier`.
3. **Runtime boundary claims** should be visible in `triton_ops.validation` and the exception types.
4. **Performance claims** should separate Benchmarking from Auto-Tuning and use Performance metrics only when shape context is known.

## Where to go next

<div class="link-grid link-grid-3">
  <a class="info-card" href="/en/academy/">
    <span class="card-kicker">Academy</span>
    <strong>System-level explanation</strong>
    <span>Read the system overview before diving into individual kernel families.</span>
  </a>
  <a class="info-card" href="/en/kernel-families/">
    <span class="card-kicker">Kernel Families</span>
    <strong>Operation-by-operation analysis</strong>
    <span>Compare fused RMSNorm + RoPE, fused Gated MLP, and the FP8 stack.</span>
  </a>
  <a class="info-card" href="/en/architecture-lab/">
    <span class="card-kicker">Architecture Lab</span>
    <strong>Module seams and runtime contracts</strong>
    <span>Use this when you need the implementation-facing explanation rather than the product-facing one.</span>
  </a>
</div>
