"""Benchmark suite for Triton operators."""

import time
from typing import Callable, Dict, List, Optional, Tuple, Any

import torch

from triton_ops.models import KernelMetrics
from triton_ops.benchmark.correctness import CorrectnessVerifier
from triton_ops.benchmark.report import BenchmarkResult, ComparisonResult, PerformanceReport
from triton_ops.autotuner.tuner import compute_gemm_metrics, compute_elementwise_metrics


class BenchmarkSuite:
    """Comprehensive benchmark suite for Triton operators.
    
    Provides functionality to:
    - Benchmark Triton kernels
    - Compare against PyTorch native operations
    - Compare against cuBLAS/cuDNN baselines
    - Verify numerical correctness
    - Generate performance reports
    
    Args:
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of benchmark iterations
        rtol: Relative tolerance for correctness
        atol: Absolute tolerance for correctness
    """
    
    def __init__(
        self,
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ):
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.verifier = CorrectnessVerifier(rtol=rtol, atol=atol)
        self.report = PerformanceReport()
        
        # Set device metadata
        if torch.cuda.is_available():
            self.report.set_metadata('device', torch.cuda.get_device_name())
            self.report.set_metadata('cuda_version', torch.version.cuda)
        self.report.set_metadata('pytorch_version', torch.__version__)
    
    def _time_kernel(
        self,
        kernel_fn: Callable,
        *args,
        **kwargs,
    ) -> float:
        """Time a kernel execution.
        
        Args:
            kernel_fn: Kernel function to time
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Average latency in milliseconds
        """
        # Warmup
        for _ in range(self.warmup_runs):
            kernel_fn(*args, **kwargs)
        
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(self.benchmark_runs):
            kernel_fn(*args, **kwargs)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        return (end - start) / self.benchmark_runs * 1000  # Convert to ms
    
    def benchmark_kernel(
        self,
        kernel_fn: Callable,
        reference_fn: Callable,
        kernel_name: str,
        problem_size: Tuple[int, ...],
        *args,
        config: Dict[str, Any] = None,
        compute_metrics_fn: Callable = None,
        **kwargs,
    ) -> BenchmarkResult:
        """Benchmark a kernel and verify correctness.
        
        Args:
            kernel_fn: Triton kernel function
            reference_fn: Reference implementation for correctness
            kernel_name: Name for reporting
            problem_size: Problem dimensions
            *args: Arguments for both functions
            config: Kernel configuration
            compute_metrics_fn: Function to compute detailed metrics
            **kwargs: Keyword arguments for both functions
            
        Returns:
            BenchmarkResult with timing and correctness info
        """
        # Get outputs
        triton_output = kernel_fn(*args, **kwargs)
        reference_output = reference_fn(*args, **kwargs)
        
        # Verify correctness
        is_correct = self.verifier.verify_allclose(triton_output, reference_output)
        
        # Time kernel
        latency_ms = self._time_kernel(kernel_fn, *args, **kwargs)
        
        # Compute metrics
        if compute_metrics_fn:
            metrics = compute_metrics_fn(problem_size, latency_ms)
        else:
            metrics = KernelMetrics(
                latency_ms=latency_ms,
                throughput_tflops=0.0,
                bandwidth_gbps=0.0,
                bandwidth_utilization=0.0,
            )
        
        result = BenchmarkResult(
            kernel_name=kernel_name,
            problem_size=problem_size,
            config=config or {},
            metrics=metrics,
            correctness=is_correct,
        )
        
        self.report.add_result(result)
        return result
    
    def compare_with_pytorch(
        self,
        triton_fn: Callable,
        pytorch_fn: Callable,
        kernel_name: str,
        problem_size: Tuple[int, ...],
        *args,
        compute_metrics_fn: Callable = None,
        **kwargs,
    ) -> ComparisonResult:
        """Compare Triton kernel with PyTorch native operation.
        
        Args:
            triton_fn: Triton kernel function
            pytorch_fn: PyTorch native function
            kernel_name: Name for reporting
            problem_size: Problem dimensions
            *args: Arguments for both functions
            compute_metrics_fn: Function to compute detailed metrics
            **kwargs: Keyword arguments
            
        Returns:
            ComparisonResult with speedup info
        """
        # Get outputs
        triton_output = triton_fn(*args, **kwargs)
        pytorch_output = pytorch_fn(*args, **kwargs)
        
        # Verify correctness
        is_correct = self.verifier.verify_allclose(triton_output, pytorch_output)
        
        # Time both
        triton_latency = self._time_kernel(triton_fn, *args, **kwargs)
        pytorch_latency = self._time_kernel(pytorch_fn, *args, **kwargs)
        
        # Compute metrics
        if compute_metrics_fn:
            triton_metrics = compute_metrics_fn(problem_size, triton_latency)
            pytorch_metrics = compute_metrics_fn(problem_size, pytorch_latency)
        else:
            triton_metrics = KernelMetrics(triton_latency, 0, 0, 0)
            pytorch_metrics = KernelMetrics(pytorch_latency, 0, 0, 0)
        
        speedup = pytorch_latency / triton_latency if triton_latency > 0 else 0
        
        comparison = ComparisonResult(
            kernel_name=kernel_name,
            problem_size=problem_size,
            triton_metrics=triton_metrics,
            baseline_metrics=pytorch_metrics,
            speedup=speedup,
            correctness=is_correct,
        )
        
        self.report.add_comparison(comparison)
        return comparison
    
    def benchmark_rmsnorm_rope(
        self,
        batch_sizes: List[int],
        seq_lens: List[int],
        hidden_dims: List[int],
        head_dim: int = 64,
    ) -> List[BenchmarkResult]:
        """Benchmark RMSNorm + RoPE across different sizes.
        
        Args:
            batch_sizes: List of batch sizes to test
            seq_lens: List of sequence lengths to test
            hidden_dims: List of hidden dimensions to test
            head_dim: Head dimension for RoPE
            
        Returns:
            List of benchmark results
        """
        from triton_ops.kernels.rmsnorm_rope import (
            fused_rmsnorm_rope,
            fused_rmsnorm_rope_reference,
        )
        
        results = []
        
        for batch in batch_sizes:
            for seq_len in seq_lens:
                for hidden_dim in hidden_dims:
                    # Create inputs
                    x = torch.randn(batch, seq_len, hidden_dim, 
                                   device='cuda', dtype=torch.float16)
                    weight = torch.randn(hidden_dim, device='cuda', dtype=torch.float16)
                    cos = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
                    sin = torch.randn(seq_len, head_dim, device='cuda', dtype=torch.float16)
                    
                    problem_size = (batch, seq_len, hidden_dim)
                    
                    def compute_metrics(size, latency):
                        numel = size[0] * size[1] * size[2]
                        return compute_elementwise_metrics(numel, latency)
                    
                    result = self.benchmark_kernel(
                        kernel_fn=fused_rmsnorm_rope,
                        reference_fn=fused_rmsnorm_rope_reference,
                        kernel_name="fused_rmsnorm_rope",
                        problem_size=problem_size,
                        x, weight, cos, sin,
                        compute_metrics_fn=compute_metrics,
                    )
                    results.append(result)
        
        return results
    
    def benchmark_gated_mlp(
        self,
        batch_sizes: List[int],
        seq_lens: List[int],
        hidden_dims: List[int],
        intermediate_dims: List[int],
        activations: List[str] = ["silu"],
    ) -> List[BenchmarkResult]:
        """Benchmark Gated MLP across different sizes.
        
        Args:
            batch_sizes: List of batch sizes
            seq_lens: List of sequence lengths
            hidden_dims: List of hidden dimensions
            intermediate_dims: List of intermediate dimensions
            activations: List of activation functions
            
        Returns:
            List of benchmark results
        """
        from triton_ops.kernels.gated_mlp import (
            fused_gated_mlp,
            gated_mlp_reference,
        )
        
        results = []
        
        for batch in batch_sizes:
            for seq_len in seq_lens:
                for hidden_dim in hidden_dims:
                    for inter_dim in intermediate_dims:
                        for activation in activations:
                            # Create inputs
                            x = torch.randn(batch, seq_len, hidden_dim,
                                          device='cuda', dtype=torch.float16)
                            gate_w = torch.randn(inter_dim, hidden_dim,
                                               device='cuda', dtype=torch.float16)
                            up_w = torch.randn(inter_dim, hidden_dim,
                                             device='cuda', dtype=torch.float16)
                            
                            problem_size = (batch, seq_len, hidden_dim, inter_dim)
                            
                            def compute_metrics(size, latency):
                                M = size[0] * size[1]
                                N = size[3]
                                K = size[2]
                                return compute_gemm_metrics(M, N, K, latency)
                            
                            result = self.benchmark_kernel(
                                kernel_fn=lambda *a, **k: fused_gated_mlp(*a, activation=activation, **k),
                                reference_fn=lambda *a, **k: gated_mlp_reference(*a, activation=activation, **k),
                                kernel_name=f"fused_gated_mlp_{activation}",
                                problem_size=problem_size,
                                x, gate_w, up_w,
                                compute_metrics_fn=compute_metrics,
                            )
                            results.append(result)
        
        return results
    
    def benchmark_fp8_gemm(
        self,
        M_sizes: List[int],
        N_sizes: List[int],
        K_sizes: List[int],
    ) -> List[BenchmarkResult]:
        """Benchmark FP8 GEMM across different sizes.
        
        Args:
            M_sizes: List of M dimensions
            N_sizes: List of N dimensions
            K_sizes: List of K dimensions
            
        Returns:
            List of benchmark results
        """
        from triton_ops.kernels.fp8_gemm import fp8_gemm, fp8_gemm_reference
        
        results = []
        
        for M in M_sizes:
            for N in N_sizes:
                for K in K_sizes:
                    # Create inputs
                    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
                    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
                    
                    problem_size = (M, N, K)
                    
                    def compute_metrics(size, latency):
                        return compute_gemm_metrics(size[0], size[1], size[2], latency)
                    
                    result = self.benchmark_kernel(
                        kernel_fn=fp8_gemm,
                        reference_fn=fp8_gemm_reference,
                        kernel_name="fp8_gemm",
                        problem_size=problem_size,
                        a, b,
                        compute_metrics_fn=compute_metrics,
                    )
                    results.append(result)
        
        return results
    
    def generate_report(self, format: str = "text") -> str:
        """Generate benchmark report.
        
        Args:
            format: "text" or "json"
            
        Returns:
            Formatted report string
        """
        if format == "json":
            return self.report.generate_json_report()
        return self.report.generate_text_report()
    
    def save_report(self, filepath: str, format: str = "text") -> None:
        """Save benchmark report to file.
        
        Args:
            filepath: Output file path
            format: "text" or "json"
        """
        self.report.save(filepath, format)
