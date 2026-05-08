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
