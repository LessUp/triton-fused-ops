---
title: Module Map
description: How triton_ops is divided into public surface, kernels, references, and support tooling
---

# Module Map

The repository is compact enough to fit in one mental model, but only if the module roles stay explicit.

## Top-level map

| Path | Role |
| :-- | :-- |
| `triton_ops/__init__.py` | Root public exports and the user-facing import surface |
| `triton_ops/kernels/` | Triton launchers, wrappers, and private implementation kernels |
| `triton_ops/reference/` | CPU/GPU reference implementations for correctness and CPU testing |
| `triton_ops/benchmark/` | Benchmark orchestration, correctness verification, and report generation |
| `triton_ops/autotuner/` | Config search, canned config sets, and cache persistence |
| `triton_ops/performance.py` | Derived Performance metrics from latency plus shape context |
| `triton_ops/models.py` | Shared dataclasses and result containers |
| `triton_ops/validation.py` | Runtime contracts and declarative validation objects |
| `triton_ops/exceptions.py` | Typed failure modes |

## Architectural intent

### `triton_ops/__init__.py`

The top-level package deliberately aggregates the stable public surface. That keeps the docs, tests, and downstream code talking about the same API surface.

### `triton_ops/kernels/`

This directory contains the implementation-specific launch paths for each Kernel family. Private `*_kernel` Triton functions live here, but they are not exported as part of the public contract.

### `triton_ops/reference/`

Reference implementations are not an afterthought. They are the proof seam for correctness checks and CPU-only regression coverage.

### `triton_ops/benchmark/` and `triton_ops/autotuner/`

These are companion subsystems, not a hidden runtime layer. Benchmarking reports evidence. Auto-Tuning searches configurations. The split keeps measurement language honest.

## Dependency heuristic

Read dependencies from outside in:

1. public import surface,
2. validation and models,
3. kernels and references,
4. benchmark and autotuner support.

That is the order that keeps the repo interview-friendly and maintainable.
