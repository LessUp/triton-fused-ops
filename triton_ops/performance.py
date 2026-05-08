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
