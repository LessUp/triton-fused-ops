"""Data models and type definitions for Triton operators."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch


@dataclass
class TensorSpec:
    """Specification for input/output tensors.
    
    Attributes:
        shape: Tuple of tensor dimensions
        dtype: PyTorch data type
        device: Device string (e.g., "cuda", "cpu")
        contiguous: Whether tensor must be contiguous
    """
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: str = "cuda"
    contiguous: bool = True
    
    def validate(self, tensor: torch.Tensor) -> bool:
        """Validate a tensor against this specification.
        
        Args:
            tensor: Tensor to validate
            
        Returns:
            True if tensor matches specification
        """
        if tensor.shape != self.shape:
            return False
        if tensor.dtype != self.dtype:
            return False
        if self.device == "cuda" and not tensor.is_cuda:
            return False
        if self.contiguous and not tensor.is_contiguous():
            return False
        return True
    
    def create_tensor(self, fill_value: Optional[float] = None) -> torch.Tensor:
        """Create a tensor matching this specification.
        
        Args:
            fill_value: Optional value to fill tensor with. If None, uses random values.
            
        Returns:
            New tensor matching specification
        """
        if fill_value is not None:
            tensor = torch.full(self.shape, fill_value, dtype=self.dtype, device=self.device)
        else:
            tensor = torch.randn(self.shape, dtype=self.dtype, device=self.device)
        return tensor


@dataclass
class RMSNormRoPEInput:
    """Input specification for fused RMSNorm + RoPE.
    
    Attributes:
        x: Input tensor spec [batch, seq_len, hidden_dim]
        weight: RMSNorm weight spec [hidden_dim]
        cos: Cosine position embeddings [seq_len, head_dim]
        sin: Sine position embeddings [seq_len, head_dim]
        eps: Small constant for numerical stability
    """
    x: TensorSpec
    weight: TensorSpec
    cos: TensorSpec
    sin: TensorSpec
    eps: float = 1e-6
    
    @classmethod
    def from_shapes(
        cls,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
        eps: float = 1e-6,
    ) -> "RMSNormRoPEInput":
        """Create input specification from dimension parameters.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            hidden_dim: Hidden dimension
            head_dim: Head dimension for RoPE
            dtype: Data type
            device: Device string
            eps: Epsilon for RMSNorm
            
        Returns:
            RMSNormRoPEInput specification
        """
        return cls(
            x=TensorSpec((batch_size, seq_len, hidden_dim), dtype, device),
            weight=TensorSpec((hidden_dim,), dtype, device),
            cos=TensorSpec((seq_len, head_dim), dtype, device),
            sin=TensorSpec((seq_len, head_dim), dtype, device),
            eps=eps,
        )


@dataclass
class GatedMLPInput:
    """Input specification for fused Gated MLP.
    
    Attributes:
        x: Input tensor spec [batch, seq_len, hidden_dim]
        gate_weight: Gate projection weight [intermediate_dim, hidden_dim]
        up_weight: Up projection weight [intermediate_dim, hidden_dim]
        activation: Activation function type ("silu" or "gelu")
    """
    x: TensorSpec
    gate_weight: TensorSpec
    up_weight: TensorSpec
    activation: Literal["silu", "gelu"] = "silu"
    
    @classmethod
    def from_shapes(
        cls,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        intermediate_dim: int,
        activation: Literal["silu", "gelu"] = "silu",
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> "GatedMLPInput":
        """Create input specification from dimension parameters.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            hidden_dim: Hidden dimension
            intermediate_dim: Intermediate (FFN) dimension
            activation: Activation function type
            dtype: Data type
            device: Device string
            
        Returns:
            GatedMLPInput specification
        """
        return cls(
            x=TensorSpec((batch_size, seq_len, hidden_dim), dtype, device),
            gate_weight=TensorSpec((intermediate_dim, hidden_dim), dtype, device),
            up_weight=TensorSpec((intermediate_dim, hidden_dim), dtype, device),
            activation=activation,
        )


@dataclass
class FP8GEMMInput:
    """Input specification for FP8 GEMM.
    
    Attributes:
        a: First matrix spec [M, K] in FP8
        b: Second matrix spec [K, N] in FP8
        a_scale: Scaling factor for A
        b_scale: Scaling factor for B
        output_dtype: Output data type (FP16 or BF16)
    """
    a: TensorSpec
    b: TensorSpec
    a_scale: TensorSpec
    b_scale: TensorSpec
    output_dtype: torch.dtype = torch.float16
    
    @classmethod
    def from_shapes(
        cls,
        M: int,
        N: int,
        K: int,
        output_dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ) -> "FP8GEMMInput":
        """Create input specification from matrix dimensions.
        
        Args:
            M: Number of rows in A and C
            N: Number of columns in B and C
            K: Number of columns in A / rows in B
            output_dtype: Output data type
            device: Device string
            
        Returns:
            FP8GEMMInput specification
        """
        # FP8 E4M3 is represented as torch.float8_e4m3fn if available, else uint8
        fp8_dtype = getattr(torch, 'float8_e4m3fn', torch.uint8)
        
        return cls(
            a=TensorSpec((M, K), fp8_dtype, device),
            b=TensorSpec((K, N), fp8_dtype, device),
            a_scale=TensorSpec((1,), torch.float32, device),
            b_scale=TensorSpec((1,), torch.float32, device),
            output_dtype=output_dtype,
        )


@dataclass
class KernelMetrics:
    """Performance metrics for a kernel execution.
    
    Attributes:
        latency_ms: Execution time in milliseconds
        throughput_tflops: Throughput in TFLOPS
        bandwidth_gbps: Memory bandwidth in GB/s
        bandwidth_utilization: Percentage of peak bandwidth utilized
    """
    latency_ms: float
    throughput_tflops: float
    bandwidth_gbps: float
    bandwidth_utilization: float
    
    def __str__(self) -> str:
        """Return human-readable string representation."""
        return (
            f"Latency: {self.latency_ms:.3f} ms, "
            f"Throughput: {self.throughput_tflops:.2f} TFLOPS, "
            f"Bandwidth: {self.bandwidth_gbps:.1f} GB/s ({self.bandwidth_utilization:.1f}%)"
        )


@dataclass
class TuningResult:
    """Result from auto-tuning.
    
    Attributes:
        best_config: Optimal configuration parameters
        metrics: Performance metrics for best configuration
        all_results: List of all tested configurations and their metrics
        problem_size: Problem size tuple used for tuning
        device: Device used for tuning
    """
    best_config: Dict[str, Any]
    metrics: KernelMetrics
    all_results: List[Tuple[Dict[str, Any], KernelMetrics]] = field(default_factory=list)
    problem_size: Optional[Tuple[int, ...]] = None
    device: Optional[str] = None
    
    def __str__(self) -> str:
        """Return human-readable string representation."""
        config_str = ", ".join(f"{k}={v}" for k, v in self.best_config.items())
        return f"Best config: {{{config_str}}}\n{self.metrics}"


@dataclass
class FP8Format:
    """FP8 E4M3 format specification.
    
    E4M3 format:
    - 1 sign bit
    - 4 exponent bits
    - 3 mantissa bits
    
    Attributes:
        exponent_bits: Number of exponent bits (4 for E4M3)
        mantissa_bits: Number of mantissa bits (3 for E4M3)
        max_value: Maximum representable value
        min_normal: Smallest normal number
    """
    exponent_bits: int = 4
    mantissa_bits: int = 3
    max_value: float = 448.0
    min_normal: float = 2**-6
    
    @staticmethod
    def compute_scale(tensor: torch.Tensor) -> torch.Tensor:
        """Compute optimal scaling factor for FP8 conversion.
        
        The scale is computed to map the tensor's range to FP8's representable range.
        
        Args:
            tensor: Input tensor to compute scale for
            
        Returns:
            Scaling factor tensor
        """
        max_abs = tensor.abs().max()
        if max_abs == 0:
            return torch.tensor(1.0, device=tensor.device, dtype=torch.float32)
        return torch.tensor(FP8Format.max_value / max_abs.item(), 
                          device=tensor.device, dtype=torch.float32)
    
    @staticmethod
    def compute_scale_per_channel(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Compute per-channel scaling factors for FP8 conversion.
        
        Args:
            tensor: Input tensor to compute scales for
            dim: Dimension along which to compute scales
            
        Returns:
            Per-channel scaling factors
        """
        max_abs = tensor.abs().amax(dim=dim, keepdim=True)
        max_abs = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs)
        return FP8Format.max_value / max_abs
    
    @staticmethod
    def is_in_range(tensor: torch.Tensor, scale: torch.Tensor) -> bool:
        """Check if scaled tensor is within FP8 representable range.
        
        Args:
            tensor: Input tensor
            scale: Scaling factor
            
        Returns:
            True if all values are within range
        """
        scaled = tensor * scale
        return scaled.abs().max() <= FP8Format.max_value
