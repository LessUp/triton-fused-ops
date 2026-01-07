#!/usr/bin/env python3
"""
Example: RMSNorm + RoPE Fusion

This example demonstrates the fused RMSNorm + Rotary Position Embedding kernel,
which combines two operations into a single GPU kernel for improved performance.

The fusion reduces memory bandwidth requirements by:
- Eliminating intermediate tensor storage
- Reducing HBM accesses from 3 to 1

Requirements:
    - CUDA-capable GPU (Ampere or newer recommended)
    - PyTorch >= 2.0
    - triton >= 2.1
"""

import time

import torch

from triton_ops import FusedRMSNormRoPE, fused_rmsnorm_rope


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """Reference RMSNorm implementation using PyTorch."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * weight


def reference_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Reference RoPE implementation using PyTorch."""
    # Assume x has shape (batch, seq_len, hidden_dim)
    # and we apply RoPE to pairs of elements
    head_dim = cos.shape[-1]
    x_rope = x[..., :head_dim]

    # Split into pairs and rotate
    x1 = x_rope[..., ::2]
    x2 = x_rope[..., 1::2]

    # Apply rotation
    cos = cos[:, : x1.shape[-1]]
    sin = sin[:, : x1.shape[-1]]

    x_rotated = torch.empty_like(x_rope)
    x_rotated[..., ::2] = x1 * cos - x2 * sin
    x_rotated[..., 1::2] = x1 * sin + x2 * cos

    # Concatenate with remaining dimensions
    if x.shape[-1] > head_dim:
        return torch.cat([x_rotated, x[..., head_dim:]], dim=-1)
    return x_rotated


def demo_functional_api():
    """Demonstrate the functional API for RMSNorm + RoPE fusion."""
    print("\n" + "=" * 60)
    print("Functional API Demo")
    print("=" * 60)

    # Configuration
    batch_size = 4
    seq_len = 512
    hidden_dim = 1024
    head_dim = 64
    eps = 1e-6

    # Create input tensors
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
    weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float16)

    # Create position embeddings (typically precomputed)
    positions = torch.arange(seq_len, device="cuda")
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device="cuda") / head_dim))
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
    cos = torch.cos(angles).to(torch.float16)
    sin = torch.sin(angles).to(torch.float16)

    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Position embedding shape: {cos.shape}")

    # Run fused kernel
    output = fused_rmsnorm_rope(x, weight, cos, sin, eps=eps)

    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")

    # Verify correctness against reference
    x_normed = reference_rmsnorm(x, weight, eps)
    reference_output = reference_rope(x_normed, cos, sin)

    # Check numerical accuracy
    max_error = torch.abs(output - reference_output).max().item()
    mean_error = torch.abs(output - reference_output).mean().item()

    print("\nNumerical Accuracy:")
    print(f"  Max absolute error: {max_error:.6f}")
    print(f"  Mean absolute error: {mean_error:.6f}")

    # Tolerance for FP16
    assert max_error < 0.01, f"Max error {max_error} exceeds tolerance"
    print("✓ Correctness verified!")


def demo_module_api():
    """Demonstrate the Module API for RMSNorm + RoPE fusion."""
    print("\n" + "=" * 60)
    print("Module API Demo")
    print("=" * 60)

    # Configuration
    hidden_dim = 1024
    head_dim = 64

    # Create module
    fused_norm = FusedRMSNormRoPE(hidden_dim, head_dim).cuda()
    print(f"Module: {fused_norm}")

    # Create inputs
    batch_size = 4
    seq_len = 512

    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)

    # Create position embeddings
    positions = torch.arange(seq_len, device="cuda")
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device="cuda") / head_dim))
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
    cos = torch.cos(angles).to(torch.float16)
    sin = torch.sin(angles).to(torch.float16)

    # Forward pass
    with torch.no_grad():
        output = fused_norm(x, cos, sin)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Module API works correctly!")


def benchmark_performance():
    """Benchmark fused vs unfused performance."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    # Configuration
    batch_size = 8
    seq_len = 2048
    hidden_dim = 4096
    head_dim = 64
    warmup_runs = 10
    benchmark_runs = 100

    # Create inputs
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
    weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float16)

    positions = torch.arange(seq_len, device="cuda")
    freqs = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device="cuda") / head_dim))
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
    cos = torch.cos(angles).to(torch.float16)
    sin = torch.sin(angles).to(torch.float16)

    print(f"Input shape: {x.shape}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Benchmark runs: {benchmark_runs}")

    # Benchmark fused kernel
    for _ in range(warmup_runs):
        _ = fused_rmsnorm_rope(x, weight, cos, sin)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(benchmark_runs):
        _ = fused_rmsnorm_rope(x, weight, cos, sin)
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / benchmark_runs * 1000

    # Benchmark unfused (reference) implementation
    for _ in range(warmup_runs):
        x_normed = reference_rmsnorm(x, weight)
        _ = reference_rope(x_normed, cos, sin)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(benchmark_runs):
        x_normed = reference_rmsnorm(x, weight)
        _ = reference_rope(x_normed, cos, sin)
    torch.cuda.synchronize()
    unfused_time = (time.perf_counter() - start) / benchmark_runs * 1000

    # Print results
    print("\nResults:")
    print(f"  Fused kernel:   {fused_time:.3f} ms")
    print(f"  Unfused (ref):  {unfused_time:.3f} ms")
    print(f"  Speedup:        {unfused_time / fused_time:.2f}x")


def main():
    """Run all demos."""
    print("=" * 60)
    print("RMSNorm + RoPE Fusion Example")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a CUDA-capable GPU.")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    demo_functional_api()
    demo_module_api()
    benchmark_performance()

    print("\n" + "=" * 60)
    print("Example completed successfully! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()
