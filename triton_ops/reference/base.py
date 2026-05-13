"""Base utilities for reference implementations.

This module provides the backend selection mechanism for reference implementations,
allowing the same function signature to work with both CPU (NumPy) and GPU (PyTorch)
backends.

The design principle:
- All reference functions accept a `backend` parameter (default: 'cpu')
- CPU backend uses NumPy for pure Python testing without GPU
- GPU backend uses PyTorch for correctness verification against Triton kernels
"""

from typing import Literal, TypeVar

import numpy as np
import torch

Backend = Literal["cpu", "cuda"]

T = TypeVar("T")


def validate_backend(backend: Backend) -> None:
    """Validate backend parameter.

    Args:
        backend: Backend to validate ('cpu' or 'cuda')

    Raises:
        ValueError: If backend is not 'cpu' or 'cuda'
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError(f"backend must be 'cpu' or 'cuda', got {backend!r}")


def ensure_numpy(tensor: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert tensor to NumPy array.

    Args:
        tensor: NumPy array or PyTorch tensor

    Returns:
        NumPy array (copy if input was PyTorch tensor)
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def ensure_torch(array: np.ndarray | torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Convert array to PyTorch tensor.

    Args:
        array: NumPy array or PyTorch tensor
        device: Target device for PyTorch tensor

    Returns:
        PyTorch tensor on specified device
    """
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).to(device=device)
    return array.to(device=device)


def to_output_dtype(
    result: np.ndarray | torch.Tensor,
    input_dtype: np.dtype | torch.dtype,
    backend: Backend,
) -> np.ndarray | torch.Tensor:
    """Convert result to match input dtype.

    Args:
        result: Result array/tensor
        input_dtype: Target dtype
        backend: Current backend

    Returns:
        Result with dtype matching input_dtype
    """
    if backend == "cpu":
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()
        if isinstance(input_dtype, torch.dtype):
            # Map torch dtype to numpy dtype
            dtype_map = {
                torch.float16: np.float16,
                torch.float32: np.float32,
                torch.bfloat16: np.float32,  # numpy doesn't have bfloat16
            }
            target_dtype = dtype_map.get(input_dtype, np.float32)
        else:
            target_dtype = input_dtype
        return result.astype(target_dtype)
    else:
        # GPU backend
        if isinstance(result, np.ndarray):
            result = torch.from_numpy(result).cuda()
        if isinstance(input_dtype, np.dtype):
            # Map numpy dtype to torch dtype
            dtype_map = {
                np.dtype(np.float16): torch.float16,
                np.dtype(np.float32): torch.float32,
            }
            target_dtype = dtype_map.get(input_dtype, torch.float32)
        else:
            target_dtype = input_dtype
        return result.to(dtype=target_dtype)
