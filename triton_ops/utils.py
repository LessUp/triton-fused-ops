"""Common utilities for Triton operators."""

import torch

from triton_ops.exceptions import DeviceError

# Minimum latency sentinel — avoids zero-division in derived metric calculations
MIN_LATENCY_MS: float = 1e-9


def sync_cuda() -> None:
    """Synchronize CUDA if available.

    This is a no-op on CPU-only systems.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def require_cuda(tensor_name: str = "input") -> None:
    """Check that CUDA is available and raise DeviceError if not.

    This is a unified helper for all Triton kernel functions.

    Args:
        tensor_name: Name of the tensor for error message

    Raises:
        DeviceError: If CUDA is not available
    """
    if not torch.cuda.is_available():
        raise DeviceError(
            "CUDA is not available. Triton kernels require a CUDA-capable GPU. "
            "For CPU testing, use triton_ops.reference module instead.",
            expected_device="cuda",
            actual_device="cpu",
            tensor_name=tensor_name,
        )


def require_tensor_on_cuda(tensor: torch.Tensor, tensor_name: str = "input") -> None:
    """Check that a tensor is on CUDA device.

    Args:
        tensor: Tensor to check
        tensor_name: Name of the tensor for error message

    Raises:
        DeviceError: If tensor is not on CUDA
    """
    if not tensor.is_cuda:
        raise DeviceError(
            f"Tensor '{tensor_name}' must be on CUDA device, but got {tensor.device}. "
            f"For CPU testing, use triton_ops.reference module instead.",
            expected_device="cuda",
            actual_device=str(tensor.device),
            tensor_name=tensor_name,
        )


def get_device_name() -> str:
    """Get the name of the current CUDA device.

    Returns:
        Device name string, or 'cpu' if CUDA is not available
    """
    if torch.cuda.is_available():
        return str(torch.cuda.get_device_name())
    return "cpu"
