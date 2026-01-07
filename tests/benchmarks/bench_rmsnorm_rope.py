"""Benchmark script for RMSNorm + RoPE kernel.

Run with: python -m tests.benchmarks.bench_rmsnorm_rope
"""

from triton_ops.benchmark import BenchmarkSuite


def main():
    """Run RMSNorm + RoPE benchmarks."""
    print("=" * 60)
    print("RMSNorm + RoPE Benchmark")
    print("=" * 60)

    suite = BenchmarkSuite(warmup_runs=10, benchmark_runs=100)

    # Test configurations
    batch_sizes = [1, 4, 8]
    seq_lens = [128, 512, 2048]
    hidden_dims = [2048, 4096]

    results = suite.benchmark_rmsnorm_rope(
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        hidden_dims=hidden_dims,
        head_dim=64,
    )

    # Print results
    print("\nResults:")
    print("-" * 60)
    for result in results:
        status = "✓" if result.correctness else "✗"
        print(f"{status} {result.kernel_name} {result.problem_size}")
        print(f"   Latency: {result.metrics.latency_ms:.3f} ms")
        print(
            f"   Bandwidth: {result.metrics.bandwidth_gbps:.1f} GB/s "
            f"({result.metrics.bandwidth_utilization:.1f}%)"
        )

    # Generate report
    report = suite.generate_report()
    print("\n" + report)

    # Save report
    suite.save_report("rmsnorm_rope_benchmark.txt", format="text")
    suite.save_report("rmsnorm_rope_benchmark.json", format="json")
    print("\nReports saved to rmsnorm_rope_benchmark.txt and .json")


if __name__ == "__main__":
    main()
