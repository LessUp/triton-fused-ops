#!/usr/bin/env python3
"""
Example: FP8 GEMM (8-bit Floating Point Matrix Multiplication)

This example demonstrates FP8 quantized matrix multiplication, which provides:
- 50% memory reduction compared to FP16
- Faster computation on supported hardware
- <1% accuracy loss for most use cases

FP8 Formats:
- E4M3: 4 exponent bits, 3 mantissa bits (better precision)
- E5M2: 5 exponent bits, 2 mantissa bits (larger dynamic range)

Requirements:
    - CUDA-capable GPU (Ampere or newer recommended)
    - PyTorch >= 2.0
    - triton >= 2.1
"""

import time

import torch

from triton_ops import FP8Linear, dequantize_fp8, fp8_gemm, quantize_fp8


def demo_basic_fp8_gemm():
    """Demonstrate basic FP8 GEMM operation."""
    print("\n" + "=" * 60)
    print("Basic FP8 GEMM Demo")
    print("=" * 60)

    # Configuration
    M, K, N = 1024, 2048, 1024

    # Create input matrices
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    print(f"Matrix A shape: {a.shape} (M x K)")
    print(f"Matrix B shape: {b.shape} (K x N)")

    # Run FP8 GEMM (auto quantization)
    output = fp8_gemm(a, b)

    print(f"Output shape: {output.shape} (M x N)")
    print(f"Output dtype: {output.dtype}")

    # Compare with FP16 baseline
    baseline = torch.matmul(a, b)

    # Calculate errors
    abs_error = torch.abs(output - baseline)
    max_error = abs_error.max().item()
    mean_error = abs_error.mean().item()
    relative_error = (abs_error / (torch.abs(baseline) + 1e-6)).mean().item()

    print("\nAccuracy vs FP16 baseline:")
    print(f"  Max absolute error: {max_error:.6f}")
    print(f"  Mean absolute error: {mean_error:.6f}")
    print(f"  Mean relative error: {relative_error * 100:.4f}%")

    print("✓ Basic FP8 GEMM completed!")


def demo_fp8_quantization():
    """Demonstrate FP8 quantization and dequantization."""
    print("\n" + "=" * 60)
    print("FP8 Quantization Demo")
    print("=" * 60)

    # Create a tensor with known distribution
    tensor = torch.randn(512, 512, device="cuda", dtype=torch.float16)

    print("Original tensor:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Min: {tensor.min().item():.4f}")
    print(f"  Max: {tensor.max().item():.4f}")
    print(f"  Mean: {tensor.mean().item():.4f}")
    print(f"  Std: {tensor.std().item():.4f}")

    # Quantize to FP8
    quantized, scale = quantize_fp8(tensor)

    print("\nQuantized tensor:")
    print(f"  Dtype: {quantized.dtype}")
    print(f"  Scale factor: {scale.item():.6f}")

    # Dequantize back
    recovered = dequantize_fp8(quantized, scale)

    print("\nRecovered tensor:")
    print(f"  Dtype: {recovered.dtype}")

    # Calculate reconstruction error
    abs_error = torch.abs(tensor - recovered)
    max_error = abs_error.max().item()
    mean_error = abs_error.mean().item()

    print("\nReconstruction accuracy:")
    print(f"  Max absolute error: {max_error:.6f}")
    print(f"  Mean absolute error: {mean_error:.6f}")

    # Check if values are close
    close_ratio = (abs_error < 0.1).float().mean().item()
    print(f"  Values within 0.1 tolerance: {close_ratio * 100:.2f}%")

    print("✓ FP8 quantization demo completed!")


def demo_fp8_linear_layer():
    """Demonstrate FP8Linear module for neural network layers."""
    print("\n" + "=" * 60)
    print("FP8Linear Module Demo")
    print("=" * 60)

    # Configuration
    in_features = 1024
    out_features = 2048
    batch_size = 8
    seq_len = 512

    # Create FP8Linear layer
    fp8_layer = FP8Linear(in_features, out_features).cuda()
    print("FP8Linear layer:")
    print(f"  In features: {in_features}")
    print(f"  Out features: {out_features}")

    # Create input
    x = torch.randn(batch_size, seq_len, in_features, device="cuda", dtype=torch.float16)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    with torch.no_grad():
        output = fp8_layer(x)

    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    # Compare with standard linear layer
    standard_layer = torch.nn.Linear(in_features, out_features, bias=False).cuda().half()
    standard_layer.weight.data = fp8_layer.weight.data.clone()

    with torch.no_grad():
        standard_output = standard_layer(x)

    # Calculate difference
    diff = torch.abs(output - standard_output).mean().item()
    print(f"\nMean difference from FP16 linear: {diff:.6f}")

    print("✓ FP8Linear module demo completed!")


def demo_memory_savings():
    """Demonstrate memory savings with FP8."""
    print("\n" + "=" * 60)
    print("Memory Savings Demo")
    print("=" * 60)

    # Configuration (simulating a large model layer)
    hidden_dim = 4096
    intermediate_dim = 11008

    # Calculate memory for different precisions
    fp32_bytes = hidden_dim * intermediate_dim * 4
    fp16_bytes = hidden_dim * intermediate_dim * 2
    fp8_bytes = hidden_dim * intermediate_dim * 1

    print(f"Weight matrix size: {hidden_dim} x {intermediate_dim}")
    print("\nMemory usage per weight matrix:")
    print(f"  FP32: {fp32_bytes / 1024 / 1024:.2f} MB")
    print(f"  FP16: {fp16_bytes / 1024 / 1024:.2f} MB")
    print(f"  FP8:  {fp8_bytes / 1024 / 1024:.2f} MB")

    print("\nMemory savings:")
    print(f"  FP8 vs FP32: {(1 - fp8_bytes / fp32_bytes) * 100:.1f}% reduction")
    print(f"  FP8 vs FP16: {(1 - fp8_bytes / fp16_bytes) * 100:.1f}% reduction")

    # For a typical LLaMA-7B model
    num_layers = 32
    num_weight_matrices = 7  # q, k, v, o, gate, up, down
    total_fp16 = num_layers * num_weight_matrices * fp16_bytes
    total_fp8 = num_layers * num_weight_matrices * fp8_bytes

    print("\nFor a 32-layer model (like LLaMA-7B):")
    print(f"  FP16 total: {total_fp16 / 1024 / 1024 / 1024:.2f} GB")
    print(f"  FP8 total:  {total_fp8 / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Savings:    {(total_fp16 - total_fp8) / 1024 / 1024 / 1024:.2f} GB")

    print("✓ Memory savings demo completed!")


def demo_accuracy_analysis():
    """Analyze FP8 accuracy across different tensor distributions."""
    print("\n" + "=" * 60)
    print("Accuracy Analysis Demo")
    print("=" * 60)

    distributions = [
        ("Normal (std=1)", lambda: torch.randn(512, 512, device="cuda", dtype=torch.float16)),
        (
            "Normal (std=0.1)",
            lambda: torch.randn(512, 512, device="cuda", dtype=torch.float16) * 0.1,
        ),
        (
            "Uniform [-1, 1]",
            lambda: torch.rand(512, 512, device="cuda", dtype=torch.float16) * 2 - 1,
        ),
        (
            "Sparse (90% zeros)",
            lambda: torch.randn(512, 512, device="cuda", dtype=torch.float16)
            * (torch.rand(512, 512, device="cuda") > 0.9).half(),
        ),
    ]

    print(f"{'Distribution':<25} {'Max Error':<12} {'Mean Error':<12} {'Rel Error':<12}")
    print("-" * 60)

    for name, gen_fn in distributions:
        tensor = gen_fn()
        quantized, scale = quantize_fp8(tensor)
        recovered = dequantize_fp8(quantized, scale)

        abs_error = torch.abs(tensor - recovered)
        max_error = abs_error.max().item()
        mean_error = abs_error.mean().item()
        rel_error = (abs_error / (torch.abs(tensor) + 1e-6)).mean().item()

        print(f"{name:<25} {max_error:<12.6f} {mean_error:<12.6f} {rel_error * 100:<11.4f}%")

    print("\n✓ Accuracy analysis completed!")


def benchmark_performance():
    """Benchmark FP8 GEMM vs FP16 GEMM."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    # Configuration
    sizes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]
    warmup_runs = 10
    benchmark_runs = 100

    print(f"{'Size (M,K,N)':<20} {'FP8 (ms)':<12} {'FP16 (ms)':<12} {'Speedup':<10}")
    print("-" * 55)

    for M, K, N in sizes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        # Warmup FP8
        for _ in range(warmup_runs):
            _ = fp8_gemm(a, b)
        torch.cuda.synchronize()

        # Benchmark FP8
        start = time.perf_counter()
        for _ in range(benchmark_runs):
            _ = fp8_gemm(a, b)
        torch.cuda.synchronize()
        fp8_time = (time.perf_counter() - start) / benchmark_runs * 1000

        # Warmup FP16
        for _ in range(warmup_runs):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

        # Benchmark FP16
        start = time.perf_counter()
        for _ in range(benchmark_runs):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        fp16_time = (time.perf_counter() - start) / benchmark_runs * 1000

        speedup = fp16_time / fp8_time
        print(f"({M},{K},{N}){'':<8} {fp8_time:<12.3f} {fp16_time:<12.3f} {speedup:<10.2f}x")

    print("\n✓ Performance benchmark completed!")


def main():
    """Run all demos."""
    print("=" * 60)
    print("FP8 GEMM Example")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a CUDA-capable GPU.")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    demo_basic_fp8_gemm()
    demo_fp8_quantization()
    demo_fp8_linear_layer()
    demo_memory_savings()
    demo_accuracy_analysis()
    benchmark_performance()

    print("\n" + "=" * 60)
    print("Example completed successfully! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()
