import pytest
import torch

import trifuse
from trifuse import performance, validation
from trifuse.exceptions import ShapeMismatchError


def test_root_module_does_not_export_shallow_runtime_helpers():
    removed_names = {
        "ContractResult",
        "GATED_MLP_CONTRACT",
        "GatedMLPInput",
        "InputContract",
        "MetricsCalculator",
        "RMSNORM_ROPE_CONTRACT",
        "RMSNormRoPEInput",
        "TensorContract",
        "require_cuda",
        "require_tensor_on_cuda",
        "validate_with_contract",
    }

    for name in removed_names:
        assert not hasattr(trifuse, name)


def test_measure_latency_synchronizes_around_benchmark_loop(monkeypatch):
    sync_events: list[str] = []
    call_count = 0

    def fake_sync() -> None:
        sync_events.append("sync")

    def kernel() -> None:
        nonlocal call_count
        call_count += 1

    monkeypatch.setattr(performance, "_sync_cuda", fake_sync)

    latency_ms = performance.measure_latency(kernel, warmup_runs=2, benchmark_runs=3)

    assert latency_ms >= 0.0
    assert call_count == 5
    assert sync_events == ["sync", "sync"]


def test_measure_metrics_applies_profile_to_measured_latency(monkeypatch):
    monkeypatch.setattr(performance, "_sync_cuda", lambda: None)

    calls = 0

    def kernel() -> None:
        nonlocal calls
        calls += 1

    metrics = performance.measure_metrics(
        kernel,
        warmup_runs=1,
        benchmark_runs=2,
        profile=performance.gemm(M=8, N=16, K=32),
    )

    assert metrics.latency_ms >= 0.0
    assert metrics.throughput_tflops > 0.0
    assert calls == 3


def test_validate_rmsnorm_rope_inputs_rejects_invalid_4d_rope_layout(monkeypatch):
    monkeypatch.setattr(validation, "_check_cuda", lambda tensor, tensor_name: None)

    x = torch.randn(2, 8, 16)
    weight = torch.randn(16)
    cos = torch.randn(2, 8, 1, 4)
    sin = torch.randn(2, 8, 1, 4)

    with pytest.raises(ShapeMismatchError, match="cos"):
        validation.validate_rmsnorm_rope_inputs(x, weight, cos, sin)
