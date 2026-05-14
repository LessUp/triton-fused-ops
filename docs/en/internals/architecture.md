---
title: Architecture
description: "How the repository is structured and how the modules relate"
---

# Architecture

The repository is organized around a small public API layer backed by validation helpers, Triton kernel implementations, performance tooling, and shared data models.

## Module map

```text
triton_ops/
├── __init__.py          # root public exports
├── performance.py       # PerformanceProfile — derived metrics seam
├── models.py            # dataclasses and metric/result containers
├── exceptions.py        # custom exception types
├── validation.py        # runtime input checks
├── utils.py             # shared helpers and constants
├── kernels/
│   ├── rmsnorm_rope.py
│   ├── gated_mlp.py
│   ├── fp8_gemm.py
│   └── fp8_quantize.py
├── reference/           # CPU/GPU reference implementations
│   ├── rmsnorm_rope.py
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

## Module dependency diagram

```mermaid
flowchart TB
    subgraph GPU["GPU Layer"]
        K1[rmsnorm_rope]
        K2[gated_mlp]
        K3[fp8_gemm]
        K4[fp8_quantize]
    end

    subgraph REF["Reference Layer"]
        R1[rmsnorm_rope]
        R2[gated_mlp]
        R3[fp8]
    end

    subgraph TOOLS["Tooling Layer"]
        T1[autotuner]
        T2[benchmark]
    end

    INIT["triton_ops.__init__"]
    VAL[validation]
    MOD[models]
    EXC[exceptions]
    PERF[performance]
    UTL[utils]

    VAL --> INIT
    MOD --> INIT
    EXC --> INIT
    UTL --> INIT
    PERF --> INIT
    PERF --> T2
    PERF --> T1

    INIT --> K1
    INIT --> K2
    INIT --> K3
    INIT --> K4

    INIT --> R1
    INIT --> R2
    INIT --> R3

    INIT --> T1
    INIT --> T2

    style GPU fill:#143,stroke:#76B900,color:#fff
    style REF fill:#124,stroke:#3476f6,color:#fff
    style TOOLS fill:#1a1a2e,stroke:#8b949e,color:#fff
    style INIT fill:#0d2600,stroke:#76B900,color:#76B900
```

> **Figure 1.** Module dependency graph. GPU-layer kernels (green) and reference implementations (blue) are independently exported from the root package. References are used for correctness verification and CPU testing but are not imported by GPU kernels at runtime. The tooling layer (gray) consumes `performance` metrics independently.

## Call chain

```mermaid
flowchart LR
    USER["User Code"] --> INIT["triton_ops.__init__"]
    INIT --> VAL["validation"]
    VAL --> LAUNCH["kernel launcher"]
    LAUNCH --> TRITON["Triton kernel"]
    TRITON --> HBM[(HBM)]

    style USER fill:#21262d,stroke:#8b949e,color:#c9d1d9
    style INIT fill:#0d2600,stroke:#76B900,color:#76B900
    style VAL fill:#1a1a2e,stroke:#ffc517,color:#ffc517
    style LAUNCH fill:#0d2600,stroke:#76B900,color:#76B900
    style TRITON fill:#143,stroke:#76B900,color:#fff
    style HBM fill:#1a1a2e,stroke:#30363d,color:#8b949e
```

> **Figure 2.** Runtime call chain. Validation (yellow) acts as a gate before any GPU work is launched. The Triton kernel reads from and writes to HBM only at the boundaries.

## Responsibility split

### Public API layer

`triton_ops.__init__` is the primary public surface. It exports kernels, module wrappers, quantization helpers, benchmark classes, autotuning tools, dataclasses, exception types, and `PerformanceProfile`. The root package is the only user-facing entry point.

### Performance metrics seam

`triton_ops.performance` provides `PerformanceProfile` objects that encapsulate problem-shape context for computing derived metrics (throughput TFLOPS, bandwidth GB/s, utilization). Used by both `BenchmarkSuite` and as an optional enrichment layer for autotuner results.

Three constructors: `latency_only()`, `elementwise(numel, ...)`, `gemm(M, N, K, ...)`.

### Reference implementation layer

`triton_ops.reference` provides reference implementations mathematically equivalent to the Triton kernels, supporting both CPU (NumPy) and GPU (PyTorch) backends. These are importable without GPU hardware and serve as:

- correctness references for kernel verification,
- test targets for unit testing without GPU (using `backend='cpu'`),
- documentation of the exact mathematical formulas.

References are exported from the root package (e.g., `reference_rmsnorm`, `reference_rope`) but are not imported by GPU kernels at runtime.

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
