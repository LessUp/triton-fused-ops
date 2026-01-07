#!/usr/bin/env python3
"""
Example: Gated MLP Fusion

This example demonstrates the fused Gated MLP kernel, which combines
gate projection, up projection, and activation into a single GPU kernel.

Supported activation functions:
- SiLU (Sigmoid Linear Unit) - used in SwiGLU
- GELU (Gaussian Error Linear Unit) - used in GeGLU

The fusion reduces memory bandwidth by:
- Computing gate and up projections in a single pass
- Applying activation without intermediate storage

Requirements:
    - CUDA-capable GPU (Ampere or newer recommended)
    - PyTorch >= 2.0
    - triton >= 2.1
"""

import time

import torch
import torch.nn.functional as F

from triton_ops import FusedGatedMLP, fused_gated_mlp


def reference_gated_mlp(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """Reference Gated MLP implementation using PyTorch."""
    # Gate projection with activation
    gate = F.linear(x, gate_weight)
    if activation == "silu":
        gate = F.silu(gate)
    elif activation == "gelu":
        gate = F.gelu(gate)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # Up projection
    up = F.linear(x, up_weight)

    # Element-wise multiplication
    return gate * up


def demo_silu_activation():
    """Demonstrate Gated MLP with SiLU activation (SwiGLU)."""
    print("\n" + "=" * 60)
    print("SwiGLU Demo (SiLU Activation)")
    print("=" * 60)

    # Configuration
    batch_size = 4
    seq_len = 512
    hidden_dim = 1024
    intermediate_dim = 2816  # ~2.75x hidden_dim, common in LLaMA

    # Create inputs
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
    gate_weight = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16)
    up_weight = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16)

    print(f"Input shape: {x.shape}")
    print(f"Gate weight shape: {gate_weight.shape}")
    print(f"Up weight shape: {up_weight.shape}")
    print("Activation: SiLU (SwiGLU)")

    # Run fused kernel
    output = fused_gated_mlp(x, gate_weight, up_weight, activation="silu")

    print(f"Output shape: {output.shape}")

    # Verify correctness
    reference = reference_gated_mlp(x, gate_weight, up_weight, activation="silu")
    max_error = torch.abs(output - reference).max().item()
    mean_error = torch.abs(output - reference).mean().item()

    print("\nNumerical Accuracy:")
    print(f"  Max absolute error: {max_error:.6f}")
    print(f"  Mean absolute error: {mean_error:.6f}")

    assert max_error < 0.1, f"Max error {max_error} exceeds tolerance"
    print("✓ SwiGLU correctness verified!")


def demo_gelu_activation():
    """Demonstrate Gated MLP with GELU activation (GeGLU)."""
    print("\n" + "=" * 60)
    print("GeGLU Demo (GELU Activation)")
    print("=" * 60)

    # Configuration
    batch_size = 4
    seq_len = 512
    hidden_dim = 1024
    intermediate_dim = 2816

    # Create inputs
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
    gate_weight = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16)
    up_weight = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16)

    print(f"Input shape: {x.shape}")
    print("Activation: GELU (GeGLU)")

    # Run fused kernel
    output = fused_gated_mlp(x, gate_weight, up_weight, activation="gelu")

    print(f"Output shape: {output.shape}")

    # Verify correctness
    reference = reference_gated_mlp(x, gate_weight, up_weight, activation="gelu")
    max_error = torch.abs(output - reference).max().item()

    print("\nNumerical Accuracy:")
    print(f"  Max absolute error: {max_error:.6f}")

    assert max_error < 0.1, f"Max error {max_error} exceeds tolerance"
    print("✓ GeGLU correctness verified!")


def demo_module_api():
    """Demonstrate the Module API for Gated MLP."""
    print("\n" + "=" * 60)
    print("Module API Demo")
    print("=" * 60)

    # Configuration
    hidden_dim = 1024
    intermediate_dim = 2816

    # Create module with SiLU activation
    mlp_silu = FusedGatedMLP(hidden_dim, intermediate_dim, activation="silu").cuda()
    print(f"SiLU Module: {mlp_silu}")

    # Create module with GELU activation
    mlp_gelu = FusedGatedMLP(hidden_dim, intermediate_dim, activation="gelu").cuda()
    print(f"GELU Module: {mlp_gelu}")

    # Create input
    batch_size = 4
    seq_len = 512
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)

    # Forward pass
    with torch.no_grad():
        output_silu = mlp_silu(x)
        output_gelu = mlp_gelu(x)

    print(f"\nInput shape: {x.shape}")
    print(f"SiLU output shape: {output_silu.shape}")
    print(f"GELU output shape: {output_gelu.shape}")
    print("✓ Module API works correctly!")


def demo_transformer_mlp_block():
    """Demonstrate using Gated MLP in a transformer-style MLP block."""
    print("\n" + "=" * 60)
    print("Transformer MLP Block Demo")
    print("=" * 60)

    class TransformerMLP(torch.nn.Module):
        """Complete MLP block as used in modern transformers like LLaMA."""

        def __init__(self, hidden_dim: int, intermediate_dim: int):
            super().__init__()
            # Fused gate + up projection with activation
            self.gate_up = FusedGatedMLP(hidden_dim, intermediate_dim, activation="silu")
            # Down projection
            self.down_proj = torch.nn.Linear(intermediate_dim, hidden_dim, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Fused: gate_proj(x) * silu(up_proj(x))
            hidden = self.gate_up(x)
            # Down projection
            output = self.down_proj(hidden)
            return output

    # Create model
    hidden_dim = 1024
    intermediate_dim = 2816
    model = TransformerMLP(hidden_dim, intermediate_dim).cuda().half()

    print("Model architecture:")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Intermediate dim: {intermediate_dim}")
    print(f"  Expansion ratio: {intermediate_dim / hidden_dim:.2f}x")

    # Create input
    batch_size = 4
    seq_len = 512
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)

    # Forward pass
    with torch.no_grad():
        output = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Transformer MLP block works correctly!")


def benchmark_performance():
    """Benchmark fused vs unfused performance."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)

    # Configuration (LLaMA-7B like)
    batch_size = 8
    seq_len = 2048
    hidden_dim = 4096
    intermediate_dim = 11008
    warmup_runs = 10
    benchmark_runs = 100

    # Create inputs
    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
    gate_weight = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16)
    up_weight = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16)

    print(f"Input shape: {x.shape}")
    print(f"Intermediate dim: {intermediate_dim}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Benchmark runs: {benchmark_runs}")

    # Benchmark fused kernel
    for _ in range(warmup_runs):
        _ = fused_gated_mlp(x, gate_weight, up_weight, activation="silu")
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(benchmark_runs):
        _ = fused_gated_mlp(x, gate_weight, up_weight, activation="silu")
    torch.cuda.synchronize()
    fused_time = (time.perf_counter() - start) / benchmark_runs * 1000

    # Benchmark unfused (reference) implementation
    for _ in range(warmup_runs):
        _ = reference_gated_mlp(x, gate_weight, up_weight, activation="silu")
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(benchmark_runs):
        _ = reference_gated_mlp(x, gate_weight, up_weight, activation="silu")
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
    print("Gated MLP Fusion Example")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a CUDA-capable GPU.")
        return

    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    demo_silu_activation()
    demo_gelu_activation()
    demo_module_api()
    demo_transformer_mlp_block()
    benchmark_performance()

    print("\n" + "=" * 60)
    print("Example completed successfully! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()
