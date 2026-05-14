"""
Performance metric helpers for Triton fused operators.

This module provides helpers for expressing kernel performance profiles
(latency-only, elementwise, and GEMM) and for computing derived metrics
(throughput, bandwidth, utilization). Calculations are intentionally
simple. MIN_LATENCY_MS is used as a zero-latency sentinel to avoid
zero-division. See repo docstring style for details.

This module serves as the single source of truth for performance metrics
computation, used by both the auto-tuner and benchmark suite.
"""

import math
from dataclasses import dataclass
from typing import Optional

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
    peak_tflops: float = (
        312.0  # Reserved for future throughput-utilization support; not used by KernelMetrics yet
    )
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
        if not (
            isinstance(self.peak_bandwidth_gbps, (int, float)) and self.peak_bandwidth_gbps > 0
        ):
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


# =============================================================================
# Unified Metrics Calculator
# =============================================================================


class MetricsCalculator:
    """Unified metrics calculator for latency → derived metrics conversion.

    This class provides a single point of computation for performance metrics,
    eliminating duplicate logic between autotuner and benchmark modules.

    The calculator can be configured with:
    - A default PerformanceProfile for all calculations
    - Optional per-call profile overrides

    Example:
        >>> calc = MetricsCalculator.default()
        >>> metrics = calc.compute(latency_ms=0.45)
        >>> print(metrics.latency_ms)
        0.45
        >>> # With a specific profile
        >>> calc = MetricsCalculator(gemm(M=1024, N=4096, K=4096))
        >>> metrics = calc.compute(latency_ms=0.45)
        >>> print(metrics.throughput_tflops)
        156.2
    """

    def __init__(self, profile: Optional[PerformanceProfile] = None):
        """Initialize the calculator with an optional profile.

        Args:
            profile: Default profile for metrics computation. If None,
                     uses latency_only() as default.
        """
        self._profile = profile if profile is not None else latency_only()

    @property
    def profile(self) -> PerformanceProfile:
        """Get the default profile."""
        return self._profile

    @profile.setter
    def profile(self, value: PerformanceProfile) -> None:
        """Set the default profile."""
        self._profile = value

    def compute(
        self,
        latency_ms: float,
        profile: Optional[PerformanceProfile] = None,
    ) -> KernelMetrics:
        """Compute metrics from latency.

        Args:
            latency_ms: Latency in milliseconds
            profile: Optional profile override. If None, uses default profile.

        Returns:
            KernelMetrics with computed values
        """
        active_profile = profile if profile is not None else self._profile
        return active_profile.metrics(latency_ms)

    @classmethod
    def default(cls) -> "MetricsCalculator":
        """Create a default calculator with latency-only profile."""
        return cls(latency_only())

    @classmethod
    def for_elementwise(
        cls,
        numel: int,
        bytes_per_element: int = 2,
        peak_bandwidth_gbps: float = 2039.0,
    ) -> "MetricsCalculator":
        """Create a calculator for elementwise operations.

        Args:
            numel: Number of elements
            bytes_per_element: Bytes per element
            peak_bandwidth_gbps: Peak memory bandwidth in GB/s

        Returns:
            Configured MetricsCalculator
        """
        return cls(
            elementwise(
                numel=numel,
                bytes_per_element=bytes_per_element,
                peak_bandwidth_gbps=peak_bandwidth_gbps,
            )
        )

    @classmethod
    def for_gemm(
        cls,
        M: int,
        N: int,
        K: int,
        bytes_per_element: int = 2,
        peak_tflops: float = 312.0,
        peak_bandwidth_gbps: float = 2039.0,
    ) -> "MetricsCalculator":
        """Create a calculator for GEMM operations.

        Args:
            M, N, K: Matrix dimensions
            bytes_per_element: Bytes per element
            peak_tflops: Peak compute throughput in TFLOPS
            peak_bandwidth_gbps: Peak memory bandwidth in GB/s

        Returns:
            Configured MetricsCalculator
        """
        return cls(
            gemm(
                M=M,
                N=N,
                K=K,
                bytes_per_element=bytes_per_element,
                peak_tflops=peak_tflops,
                peak_bandwidth_gbps=peak_bandwidth_gbps,
            )
        )


def compute_metrics(
    latency_ms: float,
    profile: Optional[PerformanceProfile] = None,
) -> KernelMetrics:
    """Convenience function for one-off metrics computation.

    This is the recommended entry point for computing metrics, used by
    both autotuner and benchmark modules.

    Args:
        latency_ms: Latency in milliseconds
        profile: Optional performance profile. If None, returns latency-only metrics.

    Returns:
        KernelMetrics with computed values

    Example:
        >>> from triton_ops.performance import compute_metrics, gemm
        >>> metrics = compute_metrics(0.45)  # latency-only
        >>> metrics = compute_metrics(0.45, profile=gemm(M=1024, N=4096, K=4096))
    """
    if profile is None:
        return latency_only().metrics(latency_ms)
    return profile.metrics(latency_ms)
