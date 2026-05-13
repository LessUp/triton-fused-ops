"""Auto-tuning framework for Triton kernels."""

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from triton_ops.autotuner.cache import ConfigCache
from triton_ops.autotuner.configs import generate_configs
from triton_ops.exceptions import TuningFailedError
from triton_ops.models import KernelMetrics, TuningResult
from triton_ops.performance import PerformanceProfile, latency_only
from triton_ops.utils import sync_cuda


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
        performance: Optional[PerformanceProfile] = None,
        **kwargs,
    ) -> Optional[KernelMetrics]:
        """Benchmark a single configuration.

        Args:
            config: Configuration to benchmark
            performance: Optional PerformanceProfile for derived metrics
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

            # Calculate latency
            total_time = end_time - start_time
            latency_ms = (total_time / self.benchmark_runs) * 1000

            # Compute metrics using PerformanceProfile
            if performance is not None:
                return performance.metrics(latency_ms)
            else:
                return latency_only().metrics(latency_ms)

        except (RuntimeError, OSError) as e:
            # Catch CUDA runtime errors and OS-level errors during kernel execution
            import warnings

            warnings.warn(f"Configuration {config} failed: {e}")
            return None

    def tune(
        self,
        *args,
        problem_size: Tuple[int, ...] = None,
        device: str = None,
        kernel_type: str = "unknown",
        performance: Optional[PerformanceProfile] = None,
        **kwargs,
    ) -> TuningResult:
        """Search configuration space and return optimal config.

        Args:
            *args: Arguments to pass to kernel
            problem_size: Problem dimensions for caching
            device: Device name for caching
            kernel_type: Kernel type identifier for caching
            performance: Optional PerformanceProfile for derived metrics
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
                metrics = self._benchmark_config(cached, *args, performance=performance, **kwargs)
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
            metrics = self._benchmark_config(config, *args, performance=performance, **kwargs)

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
