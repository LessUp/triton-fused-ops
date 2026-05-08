# Triton Fused Ops

Domain language for the repository's user-facing kernel families and its support tooling. This exists to keep architecture discussions anchored to the same concepts used in the README and docs.

## Language

**Kernel family**:
A user-facing fused operation provided by the repository, such as `fused_rmsnorm_rope`, `fused_gated_mlp`, or `fp8_gemm`.
_Avoid_: operator, primitive, workload

**Benchmarking**:
The repository tooling that verifies correctness, measures latency, and reports comparative results for a **Kernel family**.
_Avoid_: tuning, profiling harness

**Auto-Tuning**:
The repository tooling that searches configuration spaces and caches the lowest-latency configuration for a **Kernel family** or other Triton callable.
_Avoid_: benchmarking, runtime optimizer

**Performance metrics**:
Derived throughput and bandwidth numbers computed from latency plus problem-shape context for a **Kernel family**.
_Avoid_: tuning result, raw timing

## Relationships

- A **Kernel family** can be exercised by both **Benchmarking** and **Auto-Tuning**
- **Auto-Tuning** selects configurations from latency measurements
- **Benchmarking** reports **Performance metrics**
- **Performance metrics** require problem-shape context in addition to latency

## Example dialogue

> **Dev:** "Should **Auto-Tuning** keep owning the throughput formulas for a **Kernel family**?"
> **Domain expert:** "No — **Auto-Tuning** owns latency-driven configuration search, while **Performance metrics** are shared support data that **Benchmarking** can report when it has shape context."

## Flagged ambiguities

- "metrics" was being used for both raw latency from **Auto-Tuning** and derived **Performance metrics** — resolved: raw latency is part of tuning results, while throughput/bandwidth are **Performance metrics** that need shape context.
