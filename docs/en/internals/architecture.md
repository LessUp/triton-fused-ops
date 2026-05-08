---
layout: default
title: Architecture
parent: Internals
grand_parent: Documentation
nav_order: 1
description: "How the repository is structured and how the modules relate"
---

# Architecture

The repository is organized around a small public API layer backed by validation helpers, Triton kernel implementations, performance tooling, and shared data models.

## Module map

```text
triton_ops/
├── __init__.py          # root public exports
├── models.py            # dataclasses and metric/result containers
├── exceptions.py        # custom exception types
├── validation.py        # runtime input checks
├── utils.py             # shared helpers and constants
├── kernels/
│   ├── rmsnorm_rope.py
│   ├── gated_mlp.py
│   ├── fp8_gemm.py
│   └── fp8_quantize.py
├── compute/             # CPU-testable NumPy reference implementations
│   ├── rmsnorm.py
│   ├── rope.py
│   ├── gated_mlp.py
│   └── fp8.py
├── autotuner/
│   ├── configs.py
│   ├── tuner.py
│   └── cache.py
└── benchmark/
    ├── correctness.py
    ├── report.py
    └── suite.py
```

## Responsibility split

### Public API layer

`triton_ops.__init__` is the primary public surface. It exports kernels, module wrappers, quantization helpers, benchmark classes, autotuning tools, dataclasses, and exception types. The root package is the only user-facing entry point.

### Compute reference layer

`triton_ops.compute` provides pure NumPy implementations of the same mathematical operations as the Triton kernels. These are CPU-testable, importable without GPU hardware, and serve as:

- correctness references for kernel verification,
- test targets for unit testing without GPU,
- documentation of the exact mathematical formulas.

### Validation layer

`validation.py` centralizes input contracts:

- device placement,
- dtype support,
- contiguity,
- shape compatibility,
- scalar parameter checks.

This keeps the kernel entry points smaller and makes the constraints reusable in tests and wrappers.

### Kernel layer

The `kernels/` package contains the Triton implementations plus CPU/PyTorch reference implementations used for verification.

Each kernel module typically contains:

- the Triton kernel,
- the user-facing Python launcher,
- a reference function,
- an optional `nn.Module` wrapper.

### Support tooling

The autotuner and benchmark packages are separate from the kernel runtime path. They exist to support measurement, experimentation, and reporting rather than to hide tuning logic inside every API call.

## Design intent

The architecture is biased toward:

- explicit runtime contracts,
- testable reference paths,
- a small set of exported primitives,
- support code that can be reused without modifying the kernels themselves.

## Important architectural boundaries

- The repository does not ship a full transformer model stack.
- The fused kernels are building blocks intended to be embedded into larger inference code.
- Benchmarking and autotuning are companion tools, not mandatory runtime layers.
