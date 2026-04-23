"""Auto-tuning framework for Triton kernels."""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from triton_ops.autotuner.cache import ConfigCache
from triton_ops.autotuner.configs import generate_configs
from triton_ops.exceptions import TuningFailedError
from triton_ops.models import KernelMetrics, TuningResult
from triton_ops.utils import MIN_LATENCY_MS, sync_cuda


class TritonAutoTuner:
    """Auto-tuning framework for Triton kernels.

    This class provides automatic configuration search for Triton kernels,
    benchmarking different configurations and caching optimal results.

    Args:
        kernel_fn: The kernel function to tune
        config_space: Dictionary mapping parameter names to lists of values
        warmup_runs: Number of warmup runs before benchmarking
        benchmark_runs: Number of benchmark runs for timing
        cache_dir: Optional directory for persistent cache
    """

    def __init__(
        self,
        kernel_fn: Callable,
        config_space: Dict[str, List[Any]],
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
        cache_dir: Optional[str] = None,
    ):
        self.kernel_fn = kernel_fn
        self.config_space = config_space
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.cache = ConfigCache(cache_dir)

        # Generate all configurations
        self.all_configs = generate_configs(config_space)

    def _benchmark_config(
        self,
        config: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Optional[KernelMetrics]:
        """Benchmark a single configuration.

        Args:
            config: Configuration to benchmark
            *args: Arguments to pass to kernel
            **kwargs: Keyword arguments to pass to kernel

        Returns:
            KernelMetrics or None if configuration fails
        """
        try:
            # Warmup runs
            for _ in range(self.warmup_runs):
                self.kernel_fn(*args, **config, **kwargs)

            # Synchronize before timing
            sync_cuda()

            # Benchmark runs
            start_time = time.perf_counter()
            for _ in range(self.benchmark_runs):
                self.kernel_fn(*args, **config, **kwargs)
            sync_cuda()
            end_time = time.perf_counter()

            # Calculate metrics
            total_time = end_time - start_time
            latency_ms = (total_time / self.benchmark_runs) * 1000

            # Default placeholder metrics - should be computed post-tuning
            # using compute_gemm_metrics or compute_elementwise_metrics
            throughput_tflops = 0.0
            bandwidth_gbps = 0.0
            bandwidth_utilization = 0.0

            return KernelMetrics(
                latency_ms=latency_ms,
                throughput_tflops=throughput_tflops,
                bandwidth_gbps=bandwidth_gbps,
                bandwidth_utilization=bandwidth_utilization,
            )

        except Exception as e:
            import warnings

            warnings.warn(f"Configuration {config} failed: {e}")
            return None

    def tune(
        self,
        *args,
        problem_size: Tuple[int, ...] = None,
        device: str = None,
        kernel_type: str = "unknown",
        **kwargs,
    ) -> TuningResult:
        """Search configuration space and return optimal config.

        Args:
            *args: Arguments to pass to kernel
            problem_size: Problem dimensions for caching
            device: Device name for caching
            kernel_type: Kernel type identifier for caching
            **kwargs: Additional keyword arguments

        Returns:
            TuningResult with best configuration and metrics

        Raises:
            TuningFailedError: If no valid configuration found
        """
        # Check cache first
        if problem_size and device:
            cached = self.cache.get(kernel_type, problem_size, device)
            if cached:
                # Re-benchmark cached config to get metrics
                metrics = self._benchmark_config(cached, *args, **kwargs)
                if metrics:
                    return TuningResult(
                        best_config=cached,
                        metrics=metrics,
                        problem_size=problem_size,
                        device=device,
                    )

        # Benchmark all configurations
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

        # Cache result
        if problem_size and device:
            self.cache.set(kernel_type, problem_size, device, best_config)

        # best_metrics is guaranteed to be non-None here since we raise TuningFailedError above
        assert best_metrics is not None

        return TuningResult(
            best_config=best_config,
            metrics=best_metrics,
            all_results=all_results,
            problem_size=problem_size,
            device=device,
        )

    def get_cached_config(
        self,
        problem_size: Tuple[int, ...],
        device: str,
        kernel_type: str = "unknown",
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached optimal configuration.

        Args:
            problem_size: Problem dimensions
            device: Device name
            kernel_type: Kernel type identifier

        Returns:
            Cached configuration or None
        """
        return self.cache.get(kernel_type, problem_size, device)

    def clear_cache(self) -> None:
        """Clear all cached configurations."""
        self.cache.clear()


def compute_gemm_metrics(
    M: int,
    N: int,
    K: int,
    latency_ms: float,
    peak_tflops: float = 312.0,  # A100 FP16 peak
    peak_bandwidth_gbps: float = 2039.0,  # A100 HBM bandwidth
) -> KernelMetrics:
    """Compute performance metrics for GEMM operation.

    Args:
        M, N, K: Matrix dimensions
        latency_ms: Measured latency in milliseconds
        peak_tflops: Peak TFLOPS of the device
        peak_bandwidth_gbps: Peak memory bandwidth in GB/s

    Returns:
        KernelMetrics with computed values
    """
    # FLOPS for GEMM: 2 * M * N * K
    flops = 2 * M * N * K
    if latency_ms <= 0:
        latency_ms = MIN_LATENCY_MS
    tflops = flops / (latency_ms * 1e9)  # Convert to TFLOPS

    # Memory bytes: read A (M*K) + read B (K*N) + write C (M*N)
    # Assuming FP16 (2 bytes per element)
    bytes_accessed = (M * K + K * N + M * N) * 2
    bandwidth_gbps = bytes_accessed / (latency_ms * 1e6)  # Convert to GB/s

    # Utilization percentages
    throughput_utilization = (tflops / peak_tflops) * 100
    bandwidth_utilization = (bandwidth_gbps / peak_bandwidth_gbps) * 100

    return KernelMetrics(
        latency_ms=latency_ms,
        throughput_tflops=tflops,
        bandwidth_gbps=bandwidth_gbps,
        bandwidth_utilization=bandwidth_utilization,
    )


def compute_elementwise_metrics(
    numel: int,
    latency_ms: float,
    bytes_per_element: int = 2,
    peak_bandwidth_gbps: float = 2039.0,
) -> KernelMetrics:
    """Compute performance metrics for elementwise operations.

    Args:
        numel: Number of elements
        latency_ms: Measured latency in milliseconds
        bytes_per_element: Bytes per element (2 for FP16)
        peak_bandwidth_gbps: Peak memory bandwidth in GB/s

    Returns:
        KernelMetrics with computed values
    """
    # Memory bytes: read + write
    bytes_accessed = numel * bytes_per_element * 2
    if latency_ms <= 0:
        latency_ms = MIN_LATENCY_MS
    bandwidth_gbps = bytes_accessed / (latency_ms * 1e6)

    bandwidth_utilization = (bandwidth_gbps / peak_bandwidth_gbps) * 100

    return KernelMetrics(
        latency_ms=latency_ms,
        throughput_tflops=0.0,  # Not applicable for elementwise
        bandwidth_gbps=bandwidth_gbps,
        bandwidth_utilization=bandwidth_utilization,
    )
