# AGENTS.md

This repository uses an **OpenSpec-first** execution model for non-trivial changes.

## Project intent

`triton-fused-ops` is a GPU kernel library for Transformer inference: fused RMSNorm+RoPE, fused Gated MLP (SwiGLU/GeGLU), FP8 GEMM, and FP8 quantization utilities. The goal is industrial-grade stability, not feature expansion. Prioritize correctness, evidence-backed documentation, and workflow signal.

## Module vocabulary (read CONTEXT.md for full glossary)

| Term | Meaning | Do not confuse with |
|---|---|---|
| **Kernel family** | A user-facing fused op (`fused_rmsnorm_rope`, `fused_gated_mlp`, `fp8_gemm`) | Generic "kernel" or "operator" |
| **Auto-Tuning** | `triton_ops.autotuner` — config search that minimizes latency | Benchmarking |
| **Benchmarking** | `triton_ops.benchmark` — latency measurement + correctness + reports | Auto-tuning |
| **Performance metrics** | Derived throughput/bandwidth from latency + problem shape (`triton_ops.performance`) | Raw latency from tuner |
| **Compute reference** | `triton_ops.compute` — pure NumPy implementations for CPU-testable verification | Kernel implementations |

## Module map (actual structure)

```
triton_ops/
├── __init__.py           # only public surface users import from
├── kernels/              # Triton GPU implementations + nn.Module wrappers
├── compute/              # NumPy reference impls (CPU-testable, no GPU required)
├── autotuner/            # config search, caching; stays latency-focused
├── benchmark/            # suite.py, correctness.py, report.py
├── models.py             # TensorSpec, KernelMetrics, TuningResult, FP8Format
├── validation.py         # runtime input contracts
├── exceptions.py         # typed exceptions for all failure modes
└── utils.py              # shared helpers and constants
```

`triton_ops.performance` (in progress on `performance-module` branch) — shared seam between benchmarking and autotuner for derived metrics.

## Mandatory workflow (non-trivial work)

1. Create or select an OpenSpec change in `openspec/changes/`.
2. Ensure `proposal.md`, `design.md`, `tasks.md`, and required specs are complete.
3. Create a git worktree (`.worktrees/<change-name>/`) for isolation.
4. Implement tasks in order, marking checkboxes immediately.
5. Run `/review` before merge.
6. Merge to `master`; delete worktree and branch.

For OpenSpec actions: `/opsx:propose`, `/opsx:explore`, `/opsx:apply`, `/opsx:archive`

## Branch and merge policy

- **One branch per OpenSpec change.** No long-lived parallel branches.
- Default branch is `master` (not `main`).
- Merge quickly after review. Dead branches rot fast in GPU kernel repos.
- Worktrees live in `.worktrees/` (gitignored). Never commit worktree paths.

## Quality baseline (all must pass before merge)

```bash
ruff format --check .
ruff check .
mypy triton_ops/ --ignore-missing-imports
pytest tests/ -v -k "not cuda and not gpu" --ignore=tests/benchmarks/
python3 -m build
```

GPU validation on CUDA-capable hardware for kernel correctness/perf confidence (not required for CPU-only changes like `triton_ops.compute` or `triton_ops.performance`).

## Kernel API conventions

- Internal `*_kernel` Triton functions are **not exported** from `triton_ops.kernels` or root `__init__`.
- Each kernel module exposes: functional launcher (e.g. `fused_rmsnorm_rope`), `nn.Module` wrapper, reference function.
- `triton_ops.compute.*` functions are the CPU-testable counterparts to the kernel logic.
- Validation always goes through `triton_ops.validation` — never inline in kernel launchers.

## Tooling policy

- LSP baseline: **Pylance/Pyright-compatible + Ruff + mypy**.
- `mypy` override: `triton_ops.compute.*` has `warn_return_any = false` (numpy ops return `Any`).
- MCP integrations are **opt-in** and must have a clear project-specific ROI.
- Prefer long coherent sessions over high-cost parallel experimentation.
