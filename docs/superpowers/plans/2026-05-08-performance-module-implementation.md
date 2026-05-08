# Performance Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deepen the benchmark/autotuner seam by moving derived performance-metric logic into `triton_ops.performance` while preserving current benchmark behavior and a latency-first autotuner interface.

**Architecture:** Add a small top-level performance module with three constructors (`latency_only`, `elementwise`, `gemm`) that return immutable profiles capable of producing `KernelMetrics` from measured latency. Rewire `BenchmarkSuite` to depend on that seam directly, let `TritonAutoTuner.tune(...)` accept optional one-way enrichment for the winning result only, and update the English/Chinese docs to describe the new ownership model.

**Tech Stack:** Python 3.9+, PyTorch, Triton, pytest, mypy, Ruff, Jekyll docs

---

## File map

- **Create:** `triton_ops/performance.py` — the deep module that owns derived performance metrics
- **Create:** `tests/test_performance.py` — CPU-safe direct tests for the new performance test surface
- **Create:** `docs/en/api/performance.md` — English API page for the new module
- **Create:** `docs/zh/api/performance.md` — Chinese API page for the new module
- **Modify:** `triton_ops/benchmark/suite.py` — replace the autotuner leak with performance profiles and keep the fallback zero-metric path
- **Modify:** `triton_ops/autotuner/tuner.py` — keep latency-first benchmarking, remove exported metric helpers, and add optional winner-only enrichment
- **Modify:** `tests/test_benchmark.py` — add a GPU integration test proving `BenchmarkSuite` uses the new seam
- **Modify:** `tests/test_autotuner.py` — add a GPU integration test proving `tune(...)` enriches only `TuningResult.metrics`
- **Modify:** `docs/en/api/index.md` — add the performance knowledge-area card
- **Modify:** `docs/zh/api/index.md` — add the performance knowledge-area card
- **Modify:** `docs/en/api/autotuner.md` — remove the old metric-helper ownership and document the optional `performance` argument
- **Modify:** `docs/zh/api/autotuner.md` — same change in Chinese
- **Modify:** `docs/en/api/benchmark.md` — point readers at `triton_ops.performance`
- **Modify:** `docs/zh/api/benchmark.md` — same change in Chinese
- **Modify:** `docs/en/internals/architecture.md` — add `performance.py` to the module map and support-tooling split
- **Modify:** `docs/zh/internals/architecture.md` — same change in Chinese

## Task 1: Create the direct performance test surface

**Files:**
- Create: `triton_ops/performance.py`
- Test: `tests/test_performance.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from triton_ops import performance
from triton_ops.utils import MIN_LATENCY_MS


def test_latency_only_preserves_latency_and_zeroes_derived_fields():
    metrics = performance.latency_only().metrics(0.25)

    assert metrics.latency_ms == 0.25
    assert metrics.throughput_tflops == 0.0
    assert metrics.bandwidth_gbps == 0.0
    assert metrics.bandwidth_utilization == 0.0


def test_elementwise_profile_normalizes_zero_latency_and_computes_bandwidth():
    metrics = performance.elementwise(numel=256).metrics(0.0)

    expected_bandwidth = (256 * 2 * 2) / (MIN_LATENCY_MS * 1e6)
    assert metrics.latency_ms == MIN_LATENCY_MS
    assert metrics.throughput_tflops == 0.0
    assert metrics.bandwidth_gbps == pytest.approx(expected_bandwidth)
    assert metrics.bandwidth_utilization > 0.0


def test_gemm_profile_computes_throughput_and_bandwidth():
    metrics = performance.gemm(M=8, N=16, K=32).metrics(0.5)

    expected_tflops = (2 * 8 * 16 * 32) / (0.5 * 1e9)
    expected_bandwidth = ((8 * 32 + 32 * 16 + 8 * 16) * 2) / (0.5 * 1e6)
    assert metrics.throughput_tflops == pytest.approx(expected_tflops)
    assert metrics.bandwidth_gbps == pytest.approx(expected_bandwidth)


def test_invalid_profile_inputs_raise_value_error():
    with pytest.raises(ValueError):
        performance.elementwise(numel=0)

    with pytest.raises(ValueError):
        performance.gemm(M=8, N=0, K=16)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd .worktrees/performance-module
pytest tests/test_performance.py -v
```

Expected: FAIL with `ImportError` or `AttributeError` because `triton_ops.performance` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
from dataclasses import dataclass

from triton_ops.models import KernelMetrics
from triton_ops.utils import MIN_LATENCY_MS


def _normalize_latency(latency_ms: float) -> float:
    if latency_ms < 0 or latency_ms != latency_ms:
        raise ValueError("latency_ms must be a finite non-negative float")
    return latency_ms if latency_ms > 0 else MIN_LATENCY_MS


@dataclass(frozen=True)
class PerformanceProfile:
    kind: str
    dims: tuple[int, ...]
    bytes_per_element: int = 2
    peak_tflops: float = 312.0
    peak_bandwidth_gbps: float = 2039.0

    def metrics(self, latency_ms: float) -> KernelMetrics:
        latency_ms = _normalize_latency(latency_ms)

        if self.kind == "latency":
            return KernelMetrics(latency_ms, 0.0, 0.0, 0.0)

        if self.kind == "elementwise":
            (numel,) = self.dims
            bytes_accessed = numel * self.bytes_per_element * 2
            bandwidth_gbps = bytes_accessed / (latency_ms * 1e6)
            return KernelMetrics(
                latency_ms=latency_ms,
                throughput_tflops=0.0,
                bandwidth_gbps=bandwidth_gbps,
                bandwidth_utilization=(bandwidth_gbps / self.peak_bandwidth_gbps) * 100,
            )

        if self.kind == "gemm":
            M, N, K = self.dims
            flops = 2 * M * N * K
            bytes_accessed = (M * K + K * N + M * N) * self.bytes_per_element
            tflops = flops / (latency_ms * 1e9)
            bandwidth_gbps = bytes_accessed / (latency_ms * 1e6)
            return KernelMetrics(
                latency_ms=latency_ms,
                throughput_tflops=tflops,
                bandwidth_gbps=bandwidth_gbps,
                bandwidth_utilization=(bandwidth_gbps / self.peak_bandwidth_gbps) * 100,
            )

        raise ValueError(f"Unsupported performance profile kind: {self.kind}")


def latency_only() -> PerformanceProfile:
    return PerformanceProfile(kind="latency", dims=())


def elementwise(
    numel: int,
    *,
    bytes_per_element: int = 2,
    peak_bandwidth_gbps: float = 2039.0,
) -> PerformanceProfile:
    if numel <= 0 or bytes_per_element <= 0 or peak_bandwidth_gbps <= 0:
        raise ValueError("elementwise profile inputs must be positive")
    return PerformanceProfile(
        kind="elementwise",
        dims=(numel,),
        bytes_per_element=bytes_per_element,
        peak_bandwidth_gbps=peak_bandwidth_gbps,
    )


def gemm(
    *,
    M: int,
    N: int,
    K: int,
    bytes_per_element: int = 2,
    peak_tflops: float = 312.0,
    peak_bandwidth_gbps: float = 2039.0,
) -> PerformanceProfile:
    if min(M, N, K, bytes_per_element, peak_tflops, peak_bandwidth_gbps) <= 0:
        raise ValueError("gemm profile inputs must be positive")
    return PerformanceProfile(
        kind="gemm",
        dims=(M, N, K),
        bytes_per_element=bytes_per_element,
        peak_tflops=peak_tflops,
        peak_bandwidth_gbps=peak_bandwidth_gbps,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
cd .worktrees/performance-module
pytest tests/test_performance.py -v
```

Expected: PASS for all four new tests.

- [ ] **Step 5: Commit**

```bash
cd .worktrees/performance-module
git add tests/test_performance.py triton_ops/performance.py
git commit -m "feat: add shared performance module"
```

## Task 2: Rewire BenchmarkSuite to use the new seam

**Files:**
- Modify: `triton_ops/benchmark/suite.py`
- Modify: `tests/test_benchmark.py`

- [ ] **Step 1: Write the failing benchmark integration test**

```python
class TestBenchmarkSuiteIntegration:
    def test_benchmark_kernel_uses_performance_profile(self):
        from triton_ops import performance
        from triton_ops.benchmark.suite import BenchmarkSuite

        suite = BenchmarkSuite(warmup_runs=1, benchmark_runs=2)

        def triton_fn(x):
            return x * 2

        x = torch.randn(64, device="cuda", dtype=torch.float16)
        result = suite.benchmark_kernel(
            triton_fn,
            triton_fn,
            "test_kernel",
            (64,),
            x,
            performance=performance.elementwise(numel=64),
        )

        assert result.correctness
        assert result.metrics.bandwidth_gbps > 0.0
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run:

```bash
cd .worktrees/performance-module
pytest tests/test_benchmark.py::TestBenchmarkSuiteIntegration::test_benchmark_kernel_uses_performance_profile -v
```

Expected: FAIL with `TypeError: benchmark_kernel() got an unexpected keyword argument 'performance'`.

- [ ] **Step 3: Implement the BenchmarkSuite wiring**

```python
from triton_ops import performance as performance_module
from triton_ops.performance import PerformanceProfile


def benchmark_kernel(
    self,
    kernel_fn: Callable,
    reference_fn: Callable,
    kernel_name: str,
    problem_size: Tuple[int, ...],
    *args,
    config: Dict[str, Any] = None,
    performance: Optional[PerformanceProfile] = None,
    compute_metrics_fn: Callable = None,
    **kwargs,
) -> BenchmarkResult:
    triton_output = kernel_fn(*args, **kwargs)
    reference_output = reference_fn(*args, **kwargs)
    is_correct = self.verifier.verify_allclose(triton_output, reference_output)
    latency_ms = self._time_kernel(kernel_fn, *args, **kwargs)

    if performance is not None:
        metrics = performance.metrics(latency_ms)
    elif compute_metrics_fn:
        metrics = compute_metrics_fn(problem_size, latency_ms)
    else:
        metrics = KernelMetrics(latency_ms=latency_ms, throughput_tflops=0.0, bandwidth_gbps=0.0, bandwidth_utilization=0.0)

    result = BenchmarkResult(
        kernel_name=kernel_name,
        problem_size=problem_size,
        config=config or {},
        metrics=metrics,
        correctness=is_correct,
    )
    self.report.add_result(result)
    return result


def benchmark_rmsnorm_rope(
    self,
    batch_sizes: List[int],
    seq_lens: List[int],
    hidden_dims: List[int],
    head_dim: int = 64,
):
    perf = performance_module.elementwise(numel=batch * seq_len * hidden_dim)
    result = self.benchmark_kernel(
        fused_rmsnorm_rope,
        fused_rmsnorm_rope_reference,
        "fused_rmsnorm_rope",
        problem_size,
        x,
        weight,
        cos,
        sin,
        performance=perf,
    )


def benchmark_gated_mlp(
    self,
    batch_sizes: List[int],
    seq_lens: List[int],
    hidden_dims: List[int],
    intermediate_dims: List[int],
    activations: Optional[List[str]] = None,
):
    perf = performance_module.gemm(M=batch * seq_len, N=inter_dim, K=hidden_dim)
    result = self.benchmark_kernel(
        triton_fn,
        ref_fn,
        f"fused_gated_mlp_{activation}",
        problem_size,
        x,
        gate_w,
        up_w,
        performance=perf,
    )


def benchmark_fp8_gemm(
    self,
    M_sizes: List[int],
    N_sizes: List[int],
    K_sizes: List[int],
):
    perf = performance_module.gemm(M=M, N=N, K=K)
    result = self.benchmark_kernel(
        fp8_gemm,
        fp8_gemm_reference,
        "fp8_gemm",
        problem_size,
        a,
        b,
        performance=perf,
    )
```

Also remove the direct import of `compute_elementwise_metrics` / `compute_gemm_metrics` from `triton_ops.autotuner.tuner`.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run:

```bash
cd .worktrees/performance-module
pytest tests/test_performance.py tests/test_benchmark.py::TestBenchmarkSuiteIntegration::test_benchmark_kernel_uses_performance_profile -v
```

Expected: PASS, and the benchmark integration test should report non-zero bandwidth through the new seam.

- [ ] **Step 5: Commit**

```bash
cd .worktrees/performance-module
git add triton_ops/benchmark/suite.py tests/test_benchmark.py
git commit -m "refactor: route benchmark metrics through performance module"
```

## Task 3: Keep the autotuner latency-first and add winner-only enrichment

**Files:**
- Modify: `triton_ops/autotuner/tuner.py`
- Modify: `tests/test_autotuner.py`

- [ ] **Step 1: Write the failing autotuner integration test**

```python
class TestAutoTunerIntegration:
    def test_tune_enriches_winning_metrics_only(self):
        from triton_ops import performance
        from triton_ops.autotuner.tuner import TritonAutoTuner

        def dummy_kernel(*args, BLOCK_SIZE=64, num_warps=4, **kwargs):
            return torch.randn(32, device="cuda")

        tuner = TritonAutoTuner(
            kernel_fn=dummy_kernel,
            config_space={"BLOCK_SIZE": [64, 128], "num_warps": [4]},
            warmup_runs=1,
            benchmark_runs=2,
        )

        result = tuner.tune(
            problem_size=(4, 8, 16),
            device="cuda:0",
            kernel_type="dummy",
            performance=performance.gemm(M=4, N=8, K=16),
        )

        assert result.metrics.throughput_tflops > 0.0
        assert result.metrics.bandwidth_gbps > 0.0
        assert all(metrics.throughput_tflops == 0.0 for _, metrics in result.all_results)
```

- [ ] **Step 2: Run the targeted test to verify it fails**

Run:

```bash
cd .worktrees/performance-module
pytest tests/test_autotuner.py::TestAutoTunerIntegration::test_tune_enriches_winning_metrics_only -v
```

Expected: FAIL with `TypeError: tune() got an unexpected keyword argument 'performance'`.

- [ ] **Step 3: Implement winner-only enrichment and remove the old exported helpers**

```python
from triton_ops.performance import PerformanceProfile


def tune(
    self,
    *args,
    problem_size: Tuple[int, ...] = None,
    device: str = None,
    kernel_type: str = "unknown",
    performance: Optional[PerformanceProfile] = None,
    **kwargs,
) -> TuningResult:
    if problem_size and device:
        cached = self.cache.get(kernel_type, problem_size, device)
        if cached:
            metrics = self._benchmark_config(cached, *args, **kwargs)
            if metrics:
                if performance is not None:
                    metrics = performance.metrics(metrics.latency_ms)
                return TuningResult(
                    best_config=cached,
                    metrics=metrics,
                    problem_size=problem_size,
                    device=device,
                )

    all_results: List[Tuple[Dict[str, Any], KernelMetrics]] = []
    best_config = None
    best_metrics = None

    for config in self.all_configs:
        metrics = self._benchmark_config(config, *args, **kwargs)
        if metrics is not None:
            all_results.append((config.copy(), metrics))
            if best_metrics is None or metrics.latency_ms < best_metrics.latency_ms:
                best_config = config.copy()
                best_metrics = metrics

    if best_config is None:
        raise TuningFailedError(
            f"No valid configuration found for {kernel_type}",
            problem_size=problem_size,
            configs_tried=len(self.all_configs),
        )

    assert best_metrics is not None
    final_metrics = performance.metrics(best_metrics.latency_ms) if performance is not None else best_metrics
    return TuningResult(
        best_config=best_config,
        metrics=final_metrics,
        all_results=all_results,
        problem_size=problem_size,
        device=device,
    )
```

Delete `compute_gemm_metrics` and `compute_elementwise_metrics` from `triton_ops/autotuner/tuner.py` in the same commit so the new module is the only shared metric seam.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run:

```bash
cd .worktrees/performance-module
pytest tests/test_autotuner.py::TestAutoTunerIntegration::test_tune_enriches_winning_metrics_only -v
```

Expected: PASS, with enriched `result.metrics` and latency-only `all_results`.

- [ ] **Step 5: Commit**

```bash
cd .worktrees/performance-module
git add triton_ops/autotuner/tuner.py tests/test_autotuner.py
git commit -m "feat: add optional autotuner metric enrichment"
```

## Task 4: Sync the English and Chinese docs with the new ownership model

**Files:**
- Create: `docs/en/api/performance.md`
- Create: `docs/zh/api/performance.md`
- Modify: `docs/en/api/index.md`
- Modify: `docs/zh/api/index.md`
- Modify: `docs/en/api/autotuner.md`
- Modify: `docs/zh/api/autotuner.md`
- Modify: `docs/en/api/benchmark.md`
- Modify: `docs/zh/api/benchmark.md`
- Modify: `docs/en/internals/architecture.md`
- Modify: `docs/zh/internals/architecture.md`

- [ ] **Step 1: Capture the stale docs references before editing**

Run:

```bash
cd .worktrees/performance-module
rg "compute_gemm_metrics|compute_elementwise_metrics" docs/en docs/zh -n
```

Expected: matches in the benchmark/autotuner API pages that still describe the old ownership.

- [ ] **Step 2: Add the new performance API pages and update the surrounding navigation**

````md
<!-- docs/en/api/performance.md -->
---
layout: default
title: Performance
parent: API Reference
grand_parent: Documentation
nav_order: 5
description: "Derived throughput and bandwidth metrics for benchmark and autotuner tooling"
---

# Performance

`triton_ops.performance` owns derived throughput and bandwidth calculations.

```python
from triton_ops import performance

profile = performance.gemm(M=1024, N=2048, K=4096)
metrics = profile.metrics(0.45)
```

- `latency_only()`
- `elementwise(numel, bytes_per_element=2, peak_bandwidth_gbps=2039.0)`
- `gemm(M, N, K, bytes_per_element=2, peak_tflops=312.0, peak_bandwidth_gbps=2039.0)`
````

````md
<!-- docs/zh/api/performance.md -->
---
layout: default
title: 性能指标
parent: API 参考
grand_parent: 中文文档
nav_order: 5
description: "为 benchmark 与 autotuner 提供派生吞吐量/带宽指标"
---

# 性能指标

`triton_ops.performance` 负责派生吞吐量与带宽计算。

```python
from triton_ops import performance

profile = performance.gemm(M=1024, N=2048, K=4096)
metrics = profile.metrics(0.45)
```

- `latency_only()`
- `elementwise(numel, bytes_per_element=2, peak_bandwidth_gbps=2039.0)`
- `gemm(M, N, K, bytes_per_element=2, peak_tflops=312.0, peak_bandwidth_gbps=2039.0)`
````

Also make these specific edits:

- add a new “Performance” card to both API index pages
- change the autotuner card text from “search, cache, and metrics” to “search, cache, and optional enrichment”
- update the benchmark pages to point metric readers at `triton_ops.performance`
- update both architecture pages so the module map includes `performance.py` between `models.py` and `exceptions.py`

- [ ] **Step 3: Re-run the doc search and verify the old helper references are gone**

Run:

```bash
cd .worktrees/performance-module
rg "compute_gemm_metrics|compute_elementwise_metrics" docs/en docs/zh -n
```

Expected: no matches in the docs tree.

- [ ] **Step 4: Commit**

```bash
cd .worktrees/performance-module
git add \
  docs/en/api/performance.md \
  docs/zh/api/performance.md \
  docs/en/api/index.md \
  docs/zh/api/index.md \
  docs/en/api/autotuner.md \
  docs/zh/api/autotuner.md \
  docs/en/api/benchmark.md \
  docs/zh/api/benchmark.md \
  docs/en/internals/architecture.md \
  docs/zh/internals/architecture.md
git commit -m "docs: document performance module seam"
```

## Task 5: Run the repository validation suite from the isolated worktree

**Files:**
- Modify: none expected; stop and fix immediately if any validation command exposes an implementation bug

- [ ] **Step 1: Run formatting verification**

Run:

```bash
cd .worktrees/performance-module
ruff format --check .
```

Expected: exit code 0 with no files needing reformatting.

- [ ] **Step 2: Run Ruff linting**

Run:

```bash
cd .worktrees/performance-module
ruff check .
```

Expected: `All checks passed!`

- [ ] **Step 3: Run mypy over the package**

Run:

```bash
cd .worktrees/performance-module
mypy triton_ops/
```

Expected: mypy reports success with no issues.

- [ ] **Step 4: Run the CPU-safe repository test suite**

Run:

```bash
cd .worktrees/performance-module
pytest tests/ -v -k "not cuda and not gpu" --ignore=tests/benchmarks/
```

Expected: PASS for the CPU-safe suite, including the new `tests/test_performance.py`.

- [ ] **Step 5: Run the focused GPU tests for the new seam**

Run:

```bash
cd .worktrees/performance-module
pytest \
  tests/test_benchmark.py::TestBenchmarkSuiteIntegration::test_benchmark_kernel_uses_performance_profile \
  tests/test_autotuner.py::TestAutoTunerIntegration::test_tune_enriches_winning_metrics_only \
  -v
```

Expected: PASS on a CUDA-capable machine.

- [ ] **Step 6: Build the package artifacts**

Run:

```bash
cd .worktrees/performance-module
python3 -m build
```

Expected: both sdist and wheel are created successfully under `dist/`.
