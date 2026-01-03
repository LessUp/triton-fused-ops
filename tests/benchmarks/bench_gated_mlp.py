"""Benchmark script for Gated MLP kernel.

Run with: python -m tests.benchmarks.bench_gated_mlp
"""

import torch
from triton_ops.benchmark import BenchmarkSuite


def main():
    """Run Gated MLP benchmarks."""
    print("=" * 60)
    print("Gated MLP Benchmark")
    print("=" * 60)
    
    suite = BenchmarkSuite(warmup_runs=10, benchmark_runs=100)
    
    # Test configurations (typical LLaMA sizes)
    batch_sizes = [1, 4]
    seq_lens = [128, 512]
    hidden_dims = [4096]
    intermediate_dims = [11264]  # LLaMA-13B intermediate dim
    activations = ["silu", "gelu"]
    
    results = suite.benchmark_gated_mlp(
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        hidden_dims=hidden_dims,
        intermediate_dims=intermediate_dims,
        activations=activations,
    )
    
    # Print results
    print("\nResults:")
    print("-" * 60)
    for result in results:
        status = "✓" if result.correctness else "✗"
        print(f"{status} {result.kernel_name} {result.problem_size}")
        print(f"   Latency: {result.metrics.latency_ms:.3f} ms")
        print(f"   Throughput: {result.metrics.throughput_tflops:.2f} TFLOPS")
    
    # Generate report
    report = suite.generate_report()
    print("\n" + report)
    
    # Save report
    suite.save_report("gated_mlp_benchmark.txt", format="text")
    suite.save_report("gated_mlp_benchmark.json", format="json")
    print("\nReports saved to gated_mlp_benchmark.txt and .json")


if __name__ == "__main__":
    main()
