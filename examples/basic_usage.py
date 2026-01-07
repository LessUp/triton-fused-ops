#!/usr/bin/env python3
"""
Example: Basic Usage of Triton Fused Operators

This example demonstrates the basic usage of all major APIs from the
Triton Fused Operators library.

Requirements:
    - CUDA-capable GPU (Ampere or newer recommended)
    - PyTorch >= 2.0
    - triton >= 2.1
"""

import torch

# Import all major APIs from triton_ops
from triton_ops import (
    FP8Linear,
    FusedGatedMLP,
    # Module APIs
    FusedRMSNormRoPE,
    dequantize_fp8,
    fp8_gemm,
    fused_gated_mlp,
    # Functional APIs
    fused_rmsnorm_rope,
    quantize_fp8,
)


def check_cuda():
    """Check if CUDA is available."""
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a CUDA-capable GPU.")
        return False
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    return True


def demo_rmsnorm_rope():
    """Demonstrate RMSNorm + RoPE fusion."""
    print("\n" + "=" * 60)
    print("Demo: RMSNorm + RoPE Fusion")
    print("=" * 60)

    # Step 1: Prepare inputs
    batch_size = 2
    seq_len = 128
    hidden_dim = 256
    head_dim = 64

    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
    weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float16)
    cos = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
    sin = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)

    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Cos/Sin shape: {cos.shape}")

    # Step 2: Run fused operation
    output = fused_rmsnorm_rope(x, weight, cos, sin)

    # Step 3: Verify output
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output sample values: {output[0, 0, :5]}")

    print("✓ RMSNorm + RoPE fusion completed successfully!")


def demo_gated_mlp():
    """Demonstrate Gated MLP fusion."""
    print("\n" + "=" * 60)
    print("Demo: Gated MLP Fusion")
    print("=" * 60)

    # Step 1: Prepare inputs
    batch_size = 2
    seq_len = 128
    hidden_dim = 256
    intermediate_dim = 512

    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
    gate_weight = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16)
    up_weight = torch.randn(intermediate_dim, hidden_dim, device="cuda", dtype=torch.float16)

    print(f"Input shape: {x.shape}")
    print(f"Gate weight shape: {gate_weight.shape}")
    print(f"Up weight shape: {up_weight.shape}")

    # Step 2: Run with SiLU activation (SwiGLU)
    output_silu = fused_gated_mlp(x, gate_weight, up_weight, activation="silu")
    print(f"SiLU output shape: {output_silu.shape}")

    # Step 3: Run with GELU activation (GeGLU)
    output_gelu = fused_gated_mlp(x, gate_weight, up_weight, activation="gelu")
    print(f"GELU output shape: {output_gelu.shape}")

    print("✓ Gated MLP fusion completed successfully!")


def demo_fp8_gemm():
    """Demonstrate FP8 GEMM."""
    print("\n" + "=" * 60)
    print("Demo: FP8 GEMM")
    print("=" * 60)

    # Step 1: Prepare inputs
    M, K, N = 256, 512, 256

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    print(f"Matrix A shape: {a.shape}")
    print(f"Matrix B shape: {b.shape}")

    # Step 2: Run FP8 GEMM (auto quantization)
    output = fp8_gemm(a, b)

    # Step 3: Compare with PyTorch baseline
    baseline = torch.matmul(a, b)
    error = torch.abs(output - baseline).mean().item()

    print(f"Output shape: {output.shape}")
    print(f"Mean absolute error vs FP16: {error:.6f}")

    print("✓ FP8 GEMM completed successfully!")


def demo_fp8_quantization():
    """Demonstrate FP8 quantization and dequantization."""
    print("\n" + "=" * 60)
    print("Demo: FP8 Quantization")
    print("=" * 60)

    # Step 1: Prepare input tensor
    tensor = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    print(f"Original tensor shape: {tensor.shape}")
    print(f"Original tensor dtype: {tensor.dtype}")

    # Step 2: Quantize to FP8
    quantized, scale = quantize_fp8(tensor)
    print(f"Quantized tensor dtype: {quantized.dtype}")
    print(f"Scale factor: {scale.item():.6f}")

    # Step 3: Dequantize back
    recovered = dequantize_fp8(quantized, scale)
    print(f"Recovered tensor dtype: {recovered.dtype}")

    # Step 4: Check reconstruction error
    error = torch.abs(tensor - recovered).mean().item()
    print(f"Mean reconstruction error: {error:.6f}")

    print("✓ FP8 quantization completed successfully!")


def demo_module_api():
    """Demonstrate Module API usage."""
    print("\n" + "=" * 60)
    print("Demo: Module API")
    print("=" * 60)

    # Step 1: Define a simple transformer block using our modules
    class SimpleTransformerBlock(torch.nn.Module):
        def __init__(self, hidden_dim=256, head_dim=64, intermediate_dim=512):
            super().__init__()
            self.norm = FusedRMSNormRoPE(hidden_dim, head_dim)
            self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim, activation="silu")
            self.proj = FP8Linear(intermediate_dim, hidden_dim)

        def forward(self, x, cos, sin):
            x = self.norm(x, cos, sin)
            x = self.mlp(x)
            x = self.proj(x)
            return x

    # Step 2: Create model and move to GPU
    model = SimpleTransformerBlock().cuda()
    print(f"Model created: {model.__class__.__name__}")

    # Step 3: Prepare inputs
    batch_size = 2
    seq_len = 128
    hidden_dim = 256
    head_dim = 64

    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
    cos = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
    sin = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)

    # Step 4: Run forward pass
    with torch.no_grad():
        output = model(x, cos, sin)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    print("✓ Module API demo completed successfully!")


def main():
    """Run all demos."""
    print("=" * 60)
    print("Triton Fused Operators - Basic Usage Examples")
    print("=" * 60)

    # Check CUDA availability
    if not check_cuda():
        return

    # Run all demos
    demo_rmsnorm_rope()
    demo_gated_mlp()
    demo_fp8_gemm()
    demo_fp8_quantization()
    demo_module_api()

    print("\n" + "=" * 60)
    print("All demos completed successfully! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    main()
