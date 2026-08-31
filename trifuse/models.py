"""Data models and type definitions for Triton operators."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch


@dataclass
class TensorSpec:
    """Specification for input/output tensors.

    Used for input validation and tensor creation. Provides a declarative
    way to specify expected tensor properties.

    Attributes:
        shape: Tuple of tensor dimensions (e.g., ``(batch, seq_len, hidden_dim)``).
        dtype: PyTorch data type (e.g., ``torch.float16``).
        device: Device string, either ``"cuda"`` or ``"cpu"``. Defaults to ``"cuda"``.
        contiguous: Whether tensor must be contiguous in memory. Defaults to ``True``.

    Example:
        >>> spec = TensorSpec((2, 128, 4096), torch.float16, "cuda")
        >>> tensor = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
        >>> spec.validate(tensor)
        True
        >>> tensor_non_contiguous = tensor.transpose(1, 2)
        >>> spec.validate(tensor_non_contiguous)  # contiguous=True by default
        False
    """

    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: str = "cuda"
    contiguous: bool = True

    def validate(self, tensor: torch.Tensor) -> bool:
        """Validate a tensor against this specification.

        Checks that the tensor matches the expected shape, dtype, device,
        and contiguity requirements.

        Args:
            tensor: Tensor to validate.

        Returns:
            ``True`` if tensor matches all specification requirements,
            ``False`` otherwise.

        Example:
            >>> spec = TensorSpec((4, 64), torch.float16)
            >>> t = torch.randn(4, 64, device="cuda", dtype=torch.float16)
            >>> spec.validate(t)
            True
            >>> spec.validate(torch.randn(4, 32, device="cuda", dtype=torch.float16))
            False
        """
        if tensor.shape != self.shape:
            return False
        if tensor.dtype != self.dtype:
            return False
        if tensor.device.type != self.device:
            return False
        if self.contiguous and not tensor.is_contiguous():
            return False
        return True

    def create_tensor(self, fill_value: float | None = None) -> torch.Tensor:
        """Create a tensor matching this specification.

        Creates a new tensor with the specified shape, dtype, and device.
        Useful for creating test tensors or output buffers.

        Args:
            fill_value: Optional value to fill tensor with. If ``None``,
                uses random values from standard normal distribution.

        Returns:
            New tensor matching the specification.

        Example:
            >>> spec = TensorSpec((2, 4), torch.float16, "cuda")
            >>> t = spec.create_tensor(fill_value=1.0)
            >>> t.shape
            torch.Size([2, 4])
            >>> t[0, 0].item()
            1.0
        """
        if fill_value is not None:
            tensor = torch.full(self.shape, fill_value, dtype=self.dtype, device=self.device)
        else:
            tensor = torch.randn(self.shape, dtype=self.dtype, device=self.device)
        return tensor

    def validate_tensor(self, tensor: torch.Tensor) -> Tuple[bool, str | None]:
        """Validate a tensor against this specification.

        Args:
            tensor: Tensor to validate.

        Returns:
            Tuple of (is_valid, error_message).

        Example:
            >>> spec = TensorSpec((4, 64), torch.float16)
            >>> t = torch.randn(4, 64, device="cuda", dtype=torch.float16)
            >>> is_valid, error = spec.validate_tensor(t)
            >>> is_valid
            True
        """
        if tensor.shape != self.shape:
            return False, f"shape mismatch: expected {self.shape}, got {tensor.shape}"
        if tensor.dtype != self.dtype:
            return False, f"dtype mismatch: expected {self.dtype}, got {tensor.dtype}"
        if tensor.device.type != self.device:
            return False, f"device mismatch: expected {self.device}, got {tensor.device}"
        if self.contiguous and not tensor.is_contiguous():
            return False, "tensor must be contiguous"
        return True, None

    def validate_and_raise(self, tensor: torch.Tensor, name: str = "tensor") -> None:
        """Validate a tensor and raise an exception if invalid.

        Args:
            tensor: Tensor to validate.
            name: Name of the tensor for error messages.

        Raises:
            ShapeMismatchError: If shape doesn't match.
            UnsupportedDtypeError: If dtype doesn't match.
            DeviceError: If device doesn't match.
            ValueError: If contiguity doesn't match.

        Example:
            >>> spec = TensorSpec((4, 64), torch.float16)
            >>> t = torch.randn(4, 32, device="cuda", dtype=torch.float16)
            >>> spec.validate_and_raise(t, "x")  # Raises ShapeMismatchError
        """
        from trifuse.exceptions import DeviceError, ShapeMismatchError, UnsupportedDtypeError

        is_valid, error = self.validate_tensor(tensor)
        if not is_valid and error is not None:
            if "shape" in error.lower():
                raise ShapeMismatchError(f"{name}: {error}", tensor_name=name)
            elif "dtype" in error.lower():
                raise UnsupportedDtypeError(f"{name}: {error}", tensor_name=name)
            elif "device" in error.lower():
                raise DeviceError(f"{name}: {error}", tensor_name=name)
            else:
                raise ValueError(f"{name}: {error}")


@dataclass
class KernelMetrics:
    """Performance metrics for a kernel execution.

    Captures timing, throughput, and bandwidth metrics for benchmarking
    and performance analysis.

    Attributes:
        latency_ms: Execution time in milliseconds.
        throughput_tflops: Computational throughput in TFLOPS.
        bandwidth_gbps: Memory bandwidth in GB/s.
        bandwidth_utilization: Percentage of peak memory bandwidth utilized.

    Example:
        >>> metrics = KernelMetrics(
        ...     latency_ms=0.45,
        ...     throughput_tflops=156.2,
        ...     bandwidth_gbps=890.5,
        ...     bandwidth_utilization=43.7
        ... )
        >>> print(metrics)
        Latency: 0.450 ms, Throughput: 156.20 TFLOPS, Bandwidth: 890.5 GB/s (43.7%)
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
    """Result from auto-tuning configuration search.

    Contains the optimal configuration found, associated performance metrics,
    and optionally all tested configurations for analysis.

    Attributes:
        best_config: Optimal configuration dictionary with kernel parameters
            like ``BLOCK_SIZE``, ``num_warps``, ``num_stages``.
        metrics: Performance metrics for the best configuration.
        all_results: List of all tested configurations and their metrics.
            Empty if only the best result was recorded.
        problem_size: Problem dimensions used for tuning (e.g., ``(M, N, K)`` for GEMM).
        device: GPU device name used for tuning.

    Example:
        >>> result = tuner.tune(x, weight, cos, sin, problem_size=(2, 128, 4096))
        >>> result.best_config
        {'BLOCK_SIZE': 128, 'num_warps': 4}
        >>> result.metrics.latency_ms
        0.123
        >>> print(result)
        Best config: {BLOCK_SIZE=128, num_warps=4}
        Latency: 0.123 ms, Throughput: 0.00 TFLOPS, Bandwidth: 0.0 GB/s (0.0%)
    """

    best_config: Dict[str, Any]
    metrics: KernelMetrics
    all_results: List[Tuple[Dict[str, Any], KernelMetrics]] = field(default_factory=list)
    problem_size: Tuple[int, ...] | None = None
    device: str | None = None

    def __str__(self) -> str:
        """Return human-readable string representation."""
        config_str = ", ".join(f"{k}={v}" for k, v in self.best_config.items())
        return f"Best config: {{{config_str}}}\n{self.metrics}"
