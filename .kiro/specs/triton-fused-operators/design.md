# Design Document: Triton Fused Operators Library

## Overview

本设计文档描述了一套高性能 Triton 算子库的架构和实现细节。该库针对 Transformer 模型的解码阶段进行优化，通过算子融合减少 HBM 访问次数，并通过 FP8 量化提升计算吞吐量。

核心设计原则：
- **最小化内存访问**: 通过算子融合，将多次 HBM 读写合并为单次
- **最大化计算密度**: 利用 Triton 的块级并行和寄存器复用
- **灵活的精度支持**: 支持 FP32、FP16、BF16、FP8 多种精度
- **自动调优**: 通过参数搜索找到最优的 kernel 配置

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Triton Operators Library                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Fused Kernels  │  │   FP8 Kernels   │  │   Auto-Tuner    │  │
│  ├─────────────────┤  ├─────────────────┤  ├─────────────────┤  │
│  │ rmsnorm_rope    │  │ fp8_gemm        │  │ config_search   │  │
│  │ gated_mlp       │  │ fp8_quantize    │  │ benchmark       │  │
│  │                 │  │ fp8_dequantize  │  │ cache_manager   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Core Utilities                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Memory Utils    │  │ Math Primitives │  │ Validation      │  │
│  │ - block_ptr     │  │ - rsqrt         │  │ - correctness   │  │
│  │ - coalesced_io  │  │ - sigmoid       │  │ - numerical     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Python Interface                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ torch.nn.Module wrappers with autograd support              ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Fused RMSNorm + RoPE Kernel

#### Interface

```python
@triton.jit
def fused_rmsnorm_rope_kernel(
    # Input/Output pointers
    x_ptr,              # [batch, seq_len, hidden_dim]
    output_ptr,         # [batch, seq_len, hidden_dim]
    weight_ptr,         # [hidden_dim]
    cos_ptr,            # [seq_len, head_dim]
    sin_ptr,            # [seq_len, head_dim]
    # Dimensions
    batch_size,
    seq_len,
    hidden_dim,
    head_dim,
    num_heads,
    eps: tl.constexpr,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMSNorm + RoPE kernel.
    
    Step 1: RMSNorm
      - Compute variance: var = mean(x^2)
      - Normalize: x_norm = x * rsqrt(var + eps) * weight
    
    Step 2: RoPE
      - Split x_norm into pairs for rotation
      - Apply: x_rope = x_norm * cos + rotate_half(x_norm) * sin
    """
    pass
```

#### Memory Access Pattern

```
Without Fusion (3 HBM accesses):
  HBM → RMSNorm → HBM → RoPE → HBM
  
With Fusion (1 HBM access):
  HBM → [RMSNorm + RoPE in registers] → HBM
```

### 2. Fused Gated MLP Kernel

#### Interface

```python
@triton.jit
def fused_gated_mlp_kernel(
    # Input/Output pointers
    x_ptr,              # [batch, seq_len, hidden_dim]
    gate_weight_ptr,    # [intermediate_dim, hidden_dim]
    up_weight_ptr,      # [intermediate_dim, hidden_dim]
    output_ptr,         # [batch, seq_len, intermediate_dim]
    # Dimensions
    batch_size,
    seq_len,
    hidden_dim,
    intermediate_dim,
    # Activation type
    activation: tl.constexpr,  # 0=SiLU, 1=GELU
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused Gated MLP: output = gate_proj(x) * activation(up_proj(x))
    
    Fuses two matrix multiplications with element-wise activation.
    """
    pass
```

### 3. FP8 GEMM Kernel

#### Interface

```python
@triton.jit
def fp8_gemm_kernel(
    # Input pointers (FP8 format)
    a_ptr,              # [M, K] in FP8 E4M3
    b_ptr,              # [K, N] in FP8 E4M3
    c_ptr,              # [M, N] output in FP16/BF16
    # Scaling factors
    a_scale_ptr,        # Per-tensor or per-channel scale for A
    b_scale_ptr,        # Per-tensor or per-channel scale for B
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Configuration
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    FP8 Matrix Multiplication with dynamic scaling.
    
    - Inputs: FP8 E4M3 format
    - Accumulation: FP32 for numerical stability
    - Output: FP16 or BF16
    - Uses block pointers for efficient memory access
    """
    pass
```

#### FP8 Quantization Utilities

```python
@triton.jit
def quantize_fp8_kernel(
    input_ptr,          # FP16/BF16 input
    output_ptr,         # FP8 output
    scale_ptr,          # Output scale factor
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Quantize FP16/BF16 to FP8 with dynamic scaling.
    
    Algorithm:
    1. Find max absolute value in block
    2. Compute scale = max_fp8 / max_abs
    3. Scale and convert to FP8
    4. Handle overflow by adjusting scale
    """
    pass

@triton.jit
def dequantize_fp8_kernel(
    input_ptr,          # FP8 input
    output_ptr,         # FP16/BF16 output
    scale_ptr,          # Scale factor
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Dequantize FP8 back to FP16/BF16."""
    pass
```

### 4. Auto-Tuning Framework

#### Interface

```python
class TritonAutoTuner:
    """Auto-tuning framework for Triton kernels."""
    
    def __init__(
        self,
        kernel_fn: Callable,
        config_space: Dict[str, List[Any]],
        warmup_runs: int = 10,
        benchmark_runs: int = 100,
    ):
        pass
    
    def tune(
        self,
        *args,
        **kwargs,
    ) -> TuningResult:
        """
        Search configuration space and return optimal config.
        
        Returns:
            TuningResult with best_config, latency, throughput, bandwidth
        """
        pass
    
    def get_cached_config(
        self,
        problem_size: Tuple[int, ...],
        device: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached optimal configuration."""
        pass
```

#### Configuration Space

```python
RMSNORM_ROPE_CONFIGS = {
    'BLOCK_SIZE': [64, 128, 256, 512, 1024],
    'num_warps': [2, 4, 8],
    'num_stages': [1, 2, 3],
}

GATED_MLP_CONFIGS = {
    'BLOCK_M': [32, 64, 128],
    'BLOCK_N': [32, 64, 128],
    'BLOCK_K': [32, 64],
    'num_warps': [4, 8],
    'num_stages': [2, 3, 4],
}

FP8_GEMM_CONFIGS = {
    'BLOCK_M': [64, 128, 256],
    'BLOCK_N': [64, 128, 256],
    'BLOCK_K': [32, 64],
    'GROUP_SIZE_M': [4, 8],
    'num_warps': [4, 8],
    'num_stages': [3, 4, 5],
}
```

## Data Models

### Tensor Specifications

```python
from dataclasses import dataclass
from typing import Literal, Tuple
import torch

@dataclass
class TensorSpec:
    """Specification for input/output tensors."""
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: str = "cuda"
    contiguous: bool = True

@dataclass
class RMSNormRoPEInput:
    """Input specification for fused RMSNorm + RoPE."""
    x: TensorSpec           # [batch, seq_len, hidden_dim]
    weight: TensorSpec      # [hidden_dim]
    cos: TensorSpec         # [seq_len, head_dim]
    sin: TensorSpec         # [seq_len, head_dim]
    eps: float = 1e-6

@dataclass
class GatedMLPInput:
    """Input specification for fused Gated MLP."""
    x: TensorSpec           # [batch, seq_len, hidden_dim]
    gate_weight: TensorSpec # [intermediate_dim, hidden_dim]
    up_weight: TensorSpec   # [intermediate_dim, hidden_dim]
    activation: Literal["silu", "gelu"] = "silu"

@dataclass
class FP8GEMMInput:
    """Input specification for FP8 GEMM."""
    a: TensorSpec           # [M, K] in FP8
    b: TensorSpec           # [K, N] in FP8
    a_scale: TensorSpec     # Scaling factor for A
    b_scale: TensorSpec     # Scaling factor for B
    output_dtype: torch.dtype = torch.float16
```

### Performance Metrics

```python
@dataclass
class KernelMetrics:
    """Performance metrics for a kernel execution."""
    latency_ms: float
    throughput_tflops: float
    bandwidth_gbps: float
    bandwidth_utilization: float  # Percentage of peak
    
@dataclass
class TuningResult:
    """Result from auto-tuning."""
    best_config: Dict[str, Any]
    metrics: KernelMetrics
    all_results: List[Tuple[Dict[str, Any], KernelMetrics]]
```

### FP8 Format Specification

```python
@dataclass
class FP8Format:
    """FP8 E4M3 format specification."""
    exponent_bits: int = 4
    mantissa_bits: int = 3
    max_value: float = 448.0      # Max representable value
    min_normal: float = 2**-6     # Smallest normal number
    
    @staticmethod
    def compute_scale(tensor: torch.Tensor) -> torch.Tensor:
        """Compute optimal scaling factor for FP8 conversion."""
        max_abs = tensor.abs().max()
        return FP8Format.max_value / max_abs
```

## Error Handling

### Input Validation

```python
def validate_rmsnorm_rope_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> None:
    """
    Validate inputs for RMSNorm + RoPE kernel.
    
    Raises:
        ValueError: If tensor shapes are incompatible
        TypeError: If tensor dtypes are unsupported
        RuntimeError: If tensors are not on CUDA device
    """
    # Shape validation
    if x.dim() != 3:
        raise ValueError(f"Expected 3D input, got {x.dim()}D")
    
    batch, seq_len, hidden_dim = x.shape
    
    if weight.shape != (hidden_dim,):
        raise ValueError(f"Weight shape mismatch: {weight.shape} vs ({hidden_dim},)")
    
    # Device validation
    if not x.is_cuda:
        raise RuntimeError("Input tensor must be on CUDA device")
    
    # Dtype validation
    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    if x.dtype not in supported_dtypes:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
```

### Numerical Safety

```python
def handle_fp8_overflow(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    max_attempts: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Handle FP8 overflow by dynamically adjusting scale.
    
    Algorithm:
    1. Attempt quantization with current scale
    2. If overflow detected, reduce scale by factor of 2
    3. Repeat until no overflow or max attempts reached
    
    Returns:
        Tuple of (quantized_tensor, final_scale)
    """
    for attempt in range(max_attempts):
        quantized = quantize_to_fp8(tensor, scale)
        if not has_overflow(quantized):
            return quantized, scale
        scale = scale / 2.0
    
    # Final attempt with conservative scale
    return quantize_to_fp8(tensor, scale), scale
```

### Error Codes

```python
class TritonKernelError(Exception):
    """Base exception for Triton kernel errors."""
    pass

class ShapeMismatchError(TritonKernelError):
    """Raised when tensor shapes are incompatible."""
    pass

class UnsupportedDtypeError(TritonKernelError):
    """Raised when tensor dtype is not supported."""
    pass

class NumericalOverflowError(TritonKernelError):
    """Raised when numerical overflow cannot be handled."""
    pass

class TuningFailedError(TritonKernelError):
    """Raised when auto-tuning fails to find valid configuration."""
    pass
```



## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: RMSNorm + RoPE Mathematical Correctness

*For any* valid input tensor x, weight tensor w, and position embeddings (cos, sin), the fused RMSNorm + RoPE kernel output should be numerically equivalent (within floating-point tolerance) to the sequential application of:
1. RMSNorm: `x_norm = x * rsqrt(mean(x^2) + eps) * w`
2. RoPE: `x_rope = x_norm * cos + rotate_half(x_norm) * sin`

**Validates: Requirements 1.1, 1.2**

### Property 2: Gated MLP Correctness with Activation Functions

*For any* valid input tensor x, gate weights, up weights, and activation function (SiLU or GELU), the fused Gated MLP kernel output should be numerically equivalent to:
`output = gate_proj(x) * activation(up_proj(x))`

Where:
- SiLU: `silu(x) = x * sigmoid(x)`
- GELU: `gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))`

**Validates: Requirements 2.1, 2.2, 2.3**

### Property 3: Dimension Flexibility

*For any* valid combination of:
- Sequence length in range [1, 8192]
- Batch size in range [1, 64]
- Supported hidden dimensions (2048, 4096, 8192)
- Supported intermediate dimensions (5632, 11264, 22528)

The kernels should produce correct outputs without errors or crashes.

**Validates: Requirements 1.3, 1.4, 2.4, 2.5**

### Property 4: FP8 GEMM Correctness

*For any* valid FP8 matrices A and B with appropriate scaling factors, the FP8 GEMM kernel should produce a result that, when compared to FP32 reference computation, has relative error bounded by the expected FP8 precision loss.

**Validates: Requirements 3.1**

### Property 5: FP8 Quantization Round-Trip

*For any* valid FP16/BF16 tensor within the representable FP8 range, quantizing to FP8 and then dequantizing back should produce a result within the expected quantization error bounds:
`|dequantize(quantize(x)) - x| <= max_quantization_error`

**Validates: Requirements 3.4**

### Property 6: FP8 Accuracy vs FP16 Baseline

*For any* matrix multiplication problem, the FP8 GEMM result should have relative error within 1% compared to the FP16 baseline:
`|fp8_gemm(A, B) - fp16_gemm(A, B)| / |fp16_gemm(A, B)| <= 0.01`

**Validates: Requirements 3.8**

### Property 7: Auto-Tuner Cache Consistency

*For any* problem size and device configuration, if the auto-tuner has previously found an optimal configuration, retrieving the cached configuration should return the same configuration that was stored.

**Validates: Requirements 4.5**

### Property 8: Benchmark Correctness Verification

*For any* kernel implementation and reference implementation, the benchmark suite's correctness verification should correctly identify:
- Matching results (within tolerance) as correct
- Non-matching results (outside tolerance) as incorrect

**Validates: Requirements 5.3**

## Testing Strategy

### Dual Testing Approach

本项目采用双重测试策略：

1. **Unit Tests（单元测试）**: 验证特定示例、边界情况和错误条件
2. **Property-Based Tests（属性测试）**: 验证所有输入上的通用属性

两种测试互补，共同提供全面的覆盖。

### Property-Based Testing Framework

- **Framework**: [Hypothesis](https://hypothesis.readthedocs.io/) for Python
- **Minimum iterations**: 100 per property test
- **Tag format**: `Feature: triton-fused-operators, Property {number}: {property_text}`

### Test Categories

#### 1. Correctness Tests (Property-Based)

```python
# Example property test structure
@given(
    batch_size=st.integers(min_value=1, max_value=64),
    seq_len=st.integers(min_value=1, max_value=8192),
    hidden_dim=st.sampled_from([2048, 4096, 8192]),
)
@settings(max_examples=100)
def test_rmsnorm_rope_correctness(batch_size, seq_len, hidden_dim):
    """
    Feature: triton-fused-operators, Property 1: RMSNorm + RoPE Mathematical Correctness
    Validates: Requirements 1.1, 1.2
    """
    # Generate random inputs
    x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
    weight = torch.randn(hidden_dim, device='cuda')
    # ... compute reference and compare
```

#### 2. Edge Case Tests (Unit Tests)

- NaN/Inf propagation (Requirement 1.7)
- FP8 overflow handling (Requirement 3.5)
- Empty tensors
- Single-element tensors

#### 3. Integration Tests

- Single kernel launch verification (Requirements 1.5, 2.6)
- End-to-end pipeline tests

#### 4. Performance Tests (Manual Verification)

- Bandwidth utilization measurement (Requirement 1.6)
- FLOPS measurement (Requirement 3.7)
- Auto-tuning validation (Requirements 4.1-4.6)

### Test File Organization

```
tests/
├── test_rmsnorm_rope.py      # Property tests for fused RMSNorm + RoPE
├── test_gated_mlp.py         # Property tests for fused Gated MLP
├── test_fp8_gemm.py          # Property tests for FP8 GEMM
├── test_fp8_quantize.py      # Property tests for FP8 quantization
├── test_autotuner.py         # Property tests for auto-tuner
├── test_benchmark.py         # Property tests for benchmark suite
├── test_edge_cases.py        # Unit tests for edge cases
└── benchmarks/
    ├── bench_rmsnorm_rope.py
    ├── bench_gated_mlp.py
    └── bench_fp8_gemm.py
```

### Numerical Tolerance Guidelines

| Operation | Relative Tolerance | Absolute Tolerance |
|-----------|-------------------|-------------------|
| RMSNorm (FP16) | 1e-3 | 1e-5 |
| RoPE (FP16) | 1e-3 | 1e-5 |
| Gated MLP (FP16) | 1e-3 | 1e-5 |
| FP8 GEMM vs FP16 | 1e-2 | 1e-4 |
| FP8 Round-trip | 1e-2 | 1e-3 |
