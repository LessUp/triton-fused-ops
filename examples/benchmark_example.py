#!/usr/bin/env python3
"""
Example: Benchmark Suite Usage

This example demonstrates how to use the benchmark suite to:
- Measure kernel performance
- Compare against PyTorch baselines
- Generate performance reports
- Run correctness verification

Requirements:
    - CUDA-capable GPU (Ampere or newer recommended)
    - PyTorch >= 2.0
    - triton >= 2.1
"""

import time
from dataclasses import dataclass
from typing import List

import torch

from triton_ops import (
    BenchmarkSuite,
    fp8_gemm,
    fused_gated_mlp,
    fused_rmsnorm_rope,
)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    triton_time_ms: float
    pytorch_time_ms: float
    speedup: float
    max_error: float
    mean_error: float


def benchmark_single_kernel(
    name: str,
    triton_fn,
    pytorch_fn,
    inputs: tuple,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
) -> BenchmarkResult:
    """Benchmark a single kernel against PyTorch baseline."""
    # Warmup Triton
    for _ in range(warmup_runs):
        triton_output = triton_fn(*inputs)
    torch.cuda.synchronize()

    # Benchmark Triton
    start = time.perf_counter()
    for _ in range(benchmark_runs):
        triton_output = triton_fn(*inputs)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / benchmark_runs * 1000

    # Warmup PyTorch
    for _ in range(warmup_runs):
        pytorch_output = pytorch_fn(*inputs)
    torch.cuda.synchronize()

    # Benchmark PyTorch
    start = time.perf_counter()
    for _ in range(benchmark_runs):
        pytorch_output = pytorch_fn(*inputs)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / benchmark_runs * 1000

    # Calculate errors
    abs_error = torch.abs(triton_output - pytorch_output)
    max_error = abs_error.max().item()
    mean_error = abs_error.mean().item()

    return BenchmarkResult(
        name=name,
        triton_time_ms=triton_time,
        pytorch_time_ms=pytorch_time,
        speedup=pytorch_time / triton_time,
        max_error=max_error,
        mean_error=mean_error,
    )


def demo_manual_benchmarking():
    """Demonstrate manual benchmarking of individual kernels."""
    print("\n" + "=" * 60)
    print("Manual Benchmarking Demo")
    print("=" * 60)

    results: List[BenchmarkResult] = []

    # Benchmark RMSNorm + RoPE
    print("\nBenchmarking RMSNorm + RoPE...")
    batch, seq, hidden, head_dim = 8, 2048, 4096, 64

    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float16)
    weight = torch.ones(hidden, device="cuda", dtype=torch.float16)
    cos = torch.randn(seq, head_dim, device="cuda", dtype=torch.float16)
    sin = torch.randn(seq, head_dim, device="cuda", dtype=torch.float16)

    def pytorch_rmsnorm_rope(x, weight, cos, sin):
        # RMSNorm
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + 1e-6) * weight
        # Simplified RoPE (just for benchmarking)
        return x_normed

    result = benchmark_single_kernel(
        name=f"RMSNorm+RoPE ({batch}x{seq}x{hidden})",
        triton_fn=lambda x, w, c, s: fused_rmsnorm_rope(x, w, c, s),
        pytorch_fn=pytorch_rmsnorm_rope,
        inputs=(x, weight, cos, sin),
    )
    results.append(result)

    # Benchmark Gated MLP
    print("Benchmarking Gated MLP...")
    intermediate = 11008

    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float16)
    gate_w = torch.randn(intermediate, hidden, device="cuda", dtype=torch.float16)
    up_w = torch.randn(intermediate, hidden, device="cuda", dtype=torch.float16)

    def pytorch_gated_mlp(x, gate_w, up_w):
        gate = torch.nn.functional.silu(torch.nn.functional.linear(x, gate_w))
        up = torch.nn.functional.linear(x, up_w)
        return gate * up

    result = benchmark_single_kernel(
        name=f"Gated MLP ({batch}x{seq}x{hidden})",
        triton_fn=lambda x, g, u: fused_gated_mlp(x, g, u, activation="silu"),
        pytorch_fn=pytorch_gated_mlp,
        inputs=(x, gate_w, up_w),
    )
    results.append(result)

    # Benchmark FP8 GEMM
    print("Benchmarking FP8 GEMM...")
    M, K, N = 4096, 4096, 4096

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    result = benchmark_single_kernel(
        name=f"FP8 GEMM ({M}x{K}x{N})",
        triton_fn=fp8_gemm,
        pytorch_fn=torch.matmul,
        inputs=(a, b),
    )
    results.append(result)

    # Print results table
    print("\n" + "=" * 80)
    print("Benchmark Results")
    print("=" * 80)
    print(
        f"{'Kernel':<35} {'Triton (ms)':<12} {'PyTorch (ms)':<13} {'Speedup':<10} {'Max Err':<10}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r.name:<35} {r.triton_time_ms:<12.3f} {r.pytorch_time_ms:<13.3f} "
            f"{r.speedup:<10.2f}x {r.max_error:<10.6f}"
        )

    print("\n✓ Manual benchmarking completed!")


def demo_benchmark_suite():
    """Demonstrate using the BenchmarkSuite class."""
    print("\n" + "=" * 60)
    print("BenchmarkSuite Demo")
    print("=" * 60)

    # Create benchmark suite
    suite = BenchmarkSuite(warmup_runs=10, benchmark_runs=50)
    print(
        f"Created BenchmarkSuite with {suite.warmup_runs} warmup, {suite.benchmark_runs} benchmark runs"
    )

    # Run benchmarks for different configurations
    print("\nRunning RMSNorm + RoPE benchmarks...")
    rmsnorm_results = suite.benchmark_rmsnorm_rope(
        batch_sizes=[1, 4, 8],
        seq_lens=[512, 1024, 2048],
        hidden_dims=[2048, 4096],
    )

    print("\nRunning Gated MLP benchmarks...")
    mlp_results = suite.benchmark_gated_mlp(
        batch_sizes=[1, 4, 8],
        seq_lens=[512, 1024],
        hidden_dims=[2048, 4096],
    )

    # Generate and print report
    print("\n" + "=" * 60)
    print("Benchmark Report")
    print("=" * 60)

    report = suite.generate_report()
    print(report)

    # Save report to file
    report_path = "benchmark_report.txt"
    suite.save_report(report_path)
    print(f"\nReport saved to: {report_path}")

    print("\n✓ BenchmarkSuite demo completed!")


def demo_correctness_verification():
    """Demonstrate correctness verification."""
    print("\n" + "=" * 60)
    print("Correctness Verification Demo")
    print("=" * 60)

    test_cases = [
        # (name, shape_info, test_fn)
        ("RMSNorm+RoPE small", (2, 128, 256), "rmsnorm_rope"),
        ("RMSNorm+RoPE medium", (4, 512, 1024), "rmsnorm_rope"),
        ("RMSNorm+RoPE large", (8, 2048, 4096), "rmsnorm_rope"),
        ("Gated MLP small", (2, 128, 256), "gated_mlp"),
        ("Gated MLP medium", (4, 512, 1024), "gated_mlp"),
        ("FP8 GEMM small", (256, 256, 256), "fp8_gemm"),
        ("FP8 GEMM medium", (1024, 1024, 1024), "fp8_gemm"),
    ]

    print(f"{'Test Case':<25} {'Shape':<20} {'Max Error':<12} {'Status':<10}")
    print("-" * 70)

    for name, shape, test_type in test_cases:
        if test_type == "rmsnorm_rope":
            batch, seq, hidden = shape
            x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float16)
            weight = torch.ones(hidden, device="cuda", dtype=torch.float16)
            cos = torch.randn(seq, 64, device="cuda", dtype=torch.float16)
            sin = torch.randn(seq, 64, device="cuda", dtype=torch.float16)

            triton_out = fused_rmsnorm_rope(x, weight, cos, sin)

            # Reference
            variance = x.pow(2).mean(-1, keepdim=True)
            ref_out = x * torch.rsqrt(variance + 1e-6) * weight

            max_error = torch.abs(triton_out - ref_out).max().item()
            tolerance = 0.01

        elif test_type == "gated_mlp":
            batch, seq, hidden = shape
            intermediate = hidden * 2

            x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float16)
            gate_w = torch.randn(intermediate, hidden, device="cuda", dtype=torch.float16)
            up_w = torch.randn(intermediate, hidden, device="cuda", dtype=torch.float16)

            triton_out = fused_gated_mlp(x, gate_w, up_w, activation="silu")

            # Reference
            gate = torch.nn.functional.silu(torch.nn.functional.linear(x, gate_w))
            up = torch.nn.functional.linear(x, up_w)
            ref_out = gate * up

            max_error = torch.abs(triton_out - ref_out).max().item()
            tolerance = 0.1

        elif test_type == "fp8_gemm":
            M, K, N = shape
            a = torch.randn(M, K, device="cuda", dtype=torch.float16)
            b = torch.randn(K, N, device="cuda", dtype=torch.float16)

            triton_out = fp8_gemm(a, b)
            ref_out = torch.matmul(a, b)

            max_error = torch.abs(triton_out - ref_out).max().item()
            tolerance = 1.0  # FP8 has larger tolerance

        status = "✓ PASS" if max_error < tolerance else "✗ FAIL"
        shape_str = f"{shape}"
        print(f"{name:<25} {shape_str:<20} {max_error:<12.6f} {status:<10}")

    print("\n✓ Correctness verification completed!")


def demo_scaling_analysis():
    """Analyze how performance scales with problem size."""
    print("\n" + "=" * 60)
    print("Scaling Analysis Demo")
    print("=" * 60)

    # Test different sequence lengths
    batch = 4
    hidden = 4096
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]

    print(f"\nRMSNorm + RoPE scaling (batch={batch}, hidden={hidden}):")
    print(f"{'Seq Length':<12} {'Time (ms)':<12} {'Throughput (tokens/s)':<20}")
    print("-" * 45)

    for seq in seq_lengths:
        x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float16)
        weight = torch.ones(hidden, device="cuda", dtype=torch.float16)
        cos = torch.randn(seq, 64, device="cuda", dtype=torch.float16)
        sin = torch.randn(seq, 64, device="cuda", dtype=torch.float16)

        # Warmup
        for _ in range(10):
            _ = fused_rmsnorm_rope(x, weight, cos, sin)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(50):
            _ = fused_rmsnorm_rope(x, weight, cos, sin)
        torch.cuda.synchronize()
        time_ms = (time.perf_counter() - start) / 50 * 1000

        throughput = (batch * seq) / (time_ms / 1000)
        print(f"{seq:<12} {time_ms:<12.3f} {throughput:<20,.0f}")

    print("\n✓ Scaling analysis completed!")


def main():
    """Run all demos."""
    print("=" * 60)
    print("Benchmark Suite Example")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a CUDA-capable GPU.")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    demo_manual_benchmarking()
    demo_benchmark_suite()
    demo_correctness_verification()
    demo_scaling_analysis()

    print("\n" + "=" * 60)
    print("Example completed successfully! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()
