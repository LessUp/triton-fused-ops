"""Base utilities for reference implementations.

This module provides the backend selection mechanism for reference implementations,
allowing the same function signature to work with both CPU (NumPy) and GPU (PyTorch)
backends.

The design principle:
- All reference functions accept a `backend` parameter (default: 'cpu')
- CPU backend uses NumPy for pure Python testing without GPU
- GPU backend uses PyTorch for correctness verification against Triton kernels

Two approaches are supported:
1. **Manual dispatch**: Use `validate_backend`, `ensure_numpy`, `ensure_torch` directly
2. **Declarative dispatch**: Use `@reference_impl` decorator for automatic dispatch
"""

from functools import wraps
from typing import Any, Callable, Literal, TypeVar

import numpy as np
import torch

Backend = Literal["cpu", "cuda"]

T = TypeVar("T")
F = Callable[..., T]


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


# =============================================================================
# Declarative Backend Dispatcher
# =============================================================================


def reference_impl(
    cpu: Callable[..., np.ndarray] | None = None,
    cuda: Callable[..., torch.Tensor] | None = None,
) -> Callable[[F], F]:
    """Decorator for reference implementations with automatic backend dispatch.

    This decorator eliminates the boilerplate of manual backend dispatch in
    reference functions. Instead of writing:

        def foo(x, backend='cpu'):
            validate_backend(backend)
            if backend == 'cpu':
                return _foo_cpu(x)
            else:
                return _foo_cuda(x)

    You can write:

        @reference_impl(cpu=_foo_cpu, cuda=_foo_cuda)
        def foo(x, *, backend: Backend = 'cpu'):
            ...

    Args:
        cpu: CPU implementation (NumPy-based)
        cuda: CUDA implementation (PyTorch-based)

    Returns:
        Decorated function with automatic backend dispatch

    Example:
        >>> def _rmsnorm_cpu(x, weight, eps):
        ...     # NumPy implementation
        ...     ...
        >>> def _rmsnorm_cuda(x, weight, eps):
        ...     # PyTorch implementation
        ...     ...
        >>> @reference_impl(cpu=_rmsnorm_cpu, cuda=_rmsnorm_cuda)
        ... def rmsnorm(x, weight, eps=1e-6, *, backend='cpu'):
        ...     '''Reference implementation of RMSNorm.'''
        ...     pass  # Implementation provided by decorator
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, backend: Backend = "cpu", **kwargs: Any) -> Any:
            validate_backend(backend)

            if backend == "cpu":
                if cpu is None:
                    raise NotImplementedError(f"{func.__name__} does not have a CPU implementation")
                return cpu(*args, **kwargs)
            else:
                if cuda is None:
                    raise NotImplementedError(
                        f"{func.__name__} does not have a CUDA implementation"
                    )
                return cuda(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


class BackendDispatcher:
    """Class-based backend dispatcher for more complex scenarios.

    Use this when you need to:
    - Maintain state across calls
    - Share preprocessing logic between backends
    - Support runtime backend registration

    Example:
        >>> dispatcher = BackendDispatcher()
        >>> dispatcher.register('cpu', _rmsnorm_cpu)
        >>> dispatcher.register('cuda', _rmsnorm_cuda)
        >>> result = dispatcher.dispatch('cpu', x, weight, eps)
    """

    def __init__(self):
        self._implementations: dict[Backend, Callable] = {}

    def register(self, backend: Backend, impl: Callable) -> None:
        """Register an implementation for a backend."""
        self._implementations[backend] = impl

    def dispatch(self, backend: Backend, *args: Any, **kwargs: Any) -> Any:
        """Dispatch to the appropriate backend implementation."""
        validate_backend(backend)
        if backend not in self._implementations:
            raise NotImplementedError(f"No implementation registered for backend '{backend}'")
        return self._implementations[backend](*args, **kwargs)
