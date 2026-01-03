"""Benchmark script for FP8 GEMM kernel.

Run with: python -m tests.benchmarks.bench_fp8_gemm
"""

import torch
from triton_ops.benchmark import BenchmarkSuite


def main():
    """Run FP8 GEMM benchmarks."""
    print("=" * 60)
    print("FP8 GEMM Benchmark")
    print("=" * 60)
    
    suite = BenchmarkSuite(
        warmup_runs=10,
        benchmark_runs=100,
        rtol=0.05,  # Higher tolerance for FP8
        atol=1e-3,
    )
    
    # Test configurations
    M_sizes = [512, 1024, 2048]
    N_sizes = [512, 1024, 2048]
    K_sizes = [512, 1024]
    
    results = suite.benchmark_fp8_gemm(
        M_sizes=M_sizes,
        N_sizes=N_sizes,
        K_sizes=K_sizes,
    )
    
    # Print results
    print("\nResults:")
    print("-" * 60)
    for result in results:
        status = "✓" if result.correctness else "✗"
        M, N, K = result.problem_size
        print(f"{status} {result.kernel_name} M={M}, N={N}, K={K}")
        print(f"   Latency: {result.metrics.latency_ms:.3f} ms")
        print(f"   Throughput: {result.metrics.throughput_tflops:.2f} TFLOPS")
    
    # Generate report
    report = suite.generate_report()
    print("\n" + report)
    
    # Save report
    suite.save_report("fp8_gemm_benchmark.txt", format="text")
    suite.save_report("fp8_gemm_benchmark.json", format="json")
    print("\nReports saved to fp8_gemm_benchmark.txt and .json")


if __name__ == "__main__":
    main()
