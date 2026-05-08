# CLAUDE.md

Repository-specific operating guidance for Claude-based workflows on `triton-fused-ops`.

## Project essence

A focused GPU kernel library for Transformer inference (RMSNorm+RoPE, GatedMLP, FP8 GEMM). The stabilization goal: archive-ready code quality with zero outstanding correctness issues. **Expand with high proof of value or not at all.**

## Domain model (fast reference)

- `triton_ops.kernels.*` — Triton GPU implementations; each exposes a functional launcher + `nn.Module` + reference fn
- `triton_ops.compute.*` — pure NumPy CPU reference impls; test without GPU hardware
- `triton_ops.autotuner` — latency-driven config search (`TritonAutoTuner`, `ConfigCache`); stays out of throughput math
- `triton_ops.benchmark` — measurement, correctness, reporting (`BenchmarkSuite`, `CorrectnessVerifier`, `PerformanceReport`)
- `triton_ops.performance` *(in-progress branch)* — derived metrics seam shared by benchmark and autotuner
- `triton_ops.models` — `TensorSpec`, `KernelMetrics`, `TuningResult`, `FP8Format`
- `triton_ops.validation` — all runtime input contracts live here, not inline in kernels
- `triton_ops.exceptions` — `TritonKernelError`, `ShapeMismatchError`, `UnsupportedDtypeError`, etc.

## OpenSpec contract

- Start every non-trivial change from an OpenSpec artifact in `openspec/changes/`.
- Do not write production code before `proposal.md` + `design.md` + `tasks.md` are drafted.
- Mark task checkboxes immediately on completion.
- If implementation reveals scope drift, update artifacts first, then continue.
- Archive completed changes promptly: `openspec/changes/` must stay clean.

## Working principles

- Make focused, high-signal edits. No speculative refactors.
- Keep docs, CI workflows, and GitHub metadata aligned with actual code.
- `master` is the single live branch. CI triggers on `master`, not `main`.
- One worktree per change (`.worktrees/<name>/`, gitignored). Delete on merge.
- Use `/review` at integration boundaries and before merging any branch.

## Validation baseline (must all pass)

```bash
ruff format --check .
ruff check .
mypy triton_ops/ --ignore-missing-imports
pytest tests/ -v -k "not cuda and not gpu" --ignore=tests/benchmarks/
python3 -m build
```

## Key implementation decisions (don't re-debate without evidence)

- `triton_ops.compute.*` disables `mypy warn_return_any` via `pyproject.toml` override — numpy arithmetic returns `Any`, unavoidable without numpy stubs.
- Internal `*_kernel` Triton functions are private; never exported from `triton_ops.kernels` or root `__init__`.
- `triton_ops.autotuner` owns latency; `triton_ops.performance` (pending) owns derived throughput/bandwidth.
- `triton_ops.validation` centralizes all input contracts — device, dtype, contiguity, shape, scalars.

## Local overrides

Personal preferences can live in `CLAUDE.local.md`. Must not override quality gates.
