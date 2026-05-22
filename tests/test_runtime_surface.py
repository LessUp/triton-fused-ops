import pytest
import torch

import triton_ops
from triton_ops import performance, validation
from triton_ops.exceptions import DeviceError, ShapeMismatchError, UnsupportedDtypeError


def test_root_module_does_not_export_shallow_runtime_helpers():
    removed_names = {
        "ContractResult",
        "FP8GEMMInput",
        "FP8_GEMM_CONTRACT",
        "FP8_QUANTIZE_CONTRACT",
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
        assert not hasattr(triton_ops, name)


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


def test_validate_fp8_dequantize_inputs_rejects_float32_output(monkeypatch):
    monkeypatch.setattr(validation, "_check_cuda", lambda tensor, tensor_name: None)

    tensor = torch.randint(0, 255, (4,), dtype=torch.uint8)
    scale = torch.tensor(1.0, dtype=torch.float32)

    with pytest.raises(UnsupportedDtypeError, match="output_dtype"):
        validation.validate_fp8_dequantize_inputs(tensor, scale, output_dtype=torch.float32)


def test_validate_fp8_quantize_inputs_checks_tensor_and_scale_device_match(monkeypatch):
    monkeypatch.setattr(validation, "_check_cuda", lambda tensor, tensor_name: None)

    called = False

    def fake_check_same_device(*pairs):
        nonlocal called
        called = True
        raise DeviceError("device mismatch", expected_device="cuda:0", actual_device="cuda:1")

    monkeypatch.setattr(validation, "_check_same_device", fake_check_same_device)

    tensor = torch.randn(4, dtype=torch.float16)
    scale = torch.tensor(1.0, dtype=torch.float32)

    with pytest.raises(DeviceError, match="device mismatch"):
        validation.validate_fp8_quantize_inputs(tensor, scale)

    assert called
