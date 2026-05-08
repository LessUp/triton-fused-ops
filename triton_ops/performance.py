"""
Performance metric helpers for Triton fused operators.

This module provides helpers for expressing kernel performance profiles
(latency-only, elementwise, and GEMM) and for computing derived metrics
(throughput, bandwidth, utilization). Calculations are intentionally
simple. MIN_LATENCY_MS is used as a zero-latency sentinel to avoid
zero-division. See repo docstring style for details.
"""

from dataclasses import dataclass
import math

from triton_ops.models import KernelMetrics
from triton_ops.utils import MIN_LATENCY_MS


def _normalize_latency(latency_ms: float) -> float:
    """Validate and normalize latency.

    Enforce that latency is a finite, non-negative number. A zero
    latency is mapped to MIN_LATENCY_MS sentinel. +inf and -inf are rejected.
    """
    if isinstance(latency_ms, bool):
        raise ValueError("latency_ms must not be a bool")
    if not isinstance(latency_ms, (int, float)):
        raise ValueError("latency_ms must be a numeric type")
    if not math.isfinite(latency_ms) or latency_ms < 0:
        raise ValueError("latency_ms must be a finite non-negative float")
    return latency_ms if latency_ms > 0 else MIN_LATENCY_MS


@dataclass(frozen=True)
class PerformanceProfile:
    kind: str
    dims: tuple[int, ...]
    bytes_per_element: int = 2
    peak_tflops: float = 312.0  # Reserved for future throughput-utilization support; not used by KernelMetrics yet
    peak_bandwidth_gbps: float = 2039.0

    def __post_init__(self) -> None:
        valid_kinds = {"latency", "elementwise", "gemm"}
        if self.kind not in valid_kinds:
            raise ValueError(f"Unsupported performance profile kind: {self.kind!r}")

        # Invariant: bytes_per_element and peak_bandwidth_gbps must be positive
        if not (isinstance(self.bytes_per_element, int) and type(self.bytes_per_element) is int):
            raise ValueError("bytes_per_element must be a pure int, not float or bool")
        if self.bytes_per_element <= 0:
            raise ValueError("bytes_per_element must be positive")
        if not (isinstance(self.peak_bandwidth_gbps, (int, float)) and self.peak_bandwidth_gbps > 0):
            raise ValueError("peak_bandwidth_gbps must be a positive number")

        # Basic shape checks to catch typos early
        if self.kind == "latency":
            if self.dims != ():
                raise ValueError("latency profile must have empty dims")
        elif self.kind == "elementwise":
            if len(self.dims) != 1:
                raise ValueError("elementwise profile dims must be a single integer")
            numel = self.dims[0]
            if not (isinstance(numel, int) and type(numel) is int) or numel <= 0:
                raise ValueError("elementwise numel must be a positive int")
        elif self.kind == "gemm":
            if len(self.dims) != 3:
                raise ValueError("gemm profile dims must be a 3-tuple (M, N, K)")
            M, N, K = self.dims
            for name, val in ("M", M), ("N", N), ("K", K):
                if not (isinstance(val, int) and type(val) is int) or val <= 0:
                    raise ValueError(f"{name} must be a positive int for gemm profiles")

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
    # Reject non-int-like inputs explicitly
    if not (isinstance(numel, int) and type(numel) is int):
        raise ValueError("numel must be a pure int, not float or bool")
    if not (isinstance(bytes_per_element, int) and type(bytes_per_element) is int):
        raise ValueError("bytes_per_element must be a pure int, not float or bool")
    if numel <= 0:
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
    peak_tflops: float = 312.0,  # Reserved for future throughput-utilization support; not used by KernelMetrics yet
    peak_bandwidth_gbps: float = 2039.0,
) -> PerformanceProfile:
    # Reject non-int-like inputs explicitly
    for name, val in ("M", M), ("N", N), ("K", K):
        if not (isinstance(val, int) and type(val) is int):
            raise ValueError(f"{name} must be an int")
    if not (isinstance(bytes_per_element, int) and type(bytes_per_element) is int):
        raise ValueError("bytes_per_element must be a pure int, not float or bool")
    if min(M, N, K) <= 0:
        raise ValueError("gemm profile inputs must be positive")
    return PerformanceProfile(
        kind="gemm",
        dims=(M, N, K),
        bytes_per_element=bytes_per_element,
        peak_tflops=peak_tflops,
        peak_bandwidth_gbps=peak_bandwidth_gbps,
    )
