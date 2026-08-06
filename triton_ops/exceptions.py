"""Custom exceptions for Triton operators."""

from typing import Any, List, Tuple


class TritonKernelError(Exception):
    """Base exception for Triton kernel errors.

    All custom exceptions in this library inherit from this class,
    allowing users to catch all library-specific errors with a single except clause.
    """

    pass


class ShapeMismatchError(TritonKernelError):
    """Raised when tensor shapes are incompatible.

    This error is raised when input tensors have shapes that don't match
    the expected dimensions for a kernel operation.

    Attributes:
        expected: Expected shape or shape description
        actual: Actual shape received
        tensor_name: Name of the tensor with mismatched shape
    """

    def __init__(
        self,
        message: str,
        expected: Tuple[Any, ...] | None = None,
        actual: Tuple[Any, ...] | None = None,
        tensor_name: str | None = None,
    ):
        super().__init__(message)
        self.expected = expected
        self.actual = actual
        self.tensor_name = tensor_name


class UnsupportedDtypeError(TritonKernelError):
    """Raised when tensor dtype is not supported.

    This error is raised when an input tensor has a data type that
    is not supported by the kernel operation.

    Attributes:
        dtype: The unsupported dtype
        supported_dtypes: List of supported dtypes
        tensor_name: Name of the tensor with unsupported dtype
    """

    def __init__(
        self,
        message: str,
        dtype: Any = None,
        supported_dtypes: List[Any] | None = None,
        tensor_name: str | None = None,
    ):
        super().__init__(message)
        self.dtype = dtype
        self.supported_dtypes = supported_dtypes
        self.tensor_name = tensor_name


class TuningFailedError(TritonKernelError):
    """Raised when auto-tuning fails to find valid configuration.

    This error is raised when the auto-tuner cannot find any valid
    configuration for the given problem size and hardware.

    Attributes:
        problem_size: Problem size that failed tuning
        configs_tried: Number of configurations attempted
        last_error: Last error encountered during tuning
    """

    def __init__(
        self,
        message: str,
        problem_size: Tuple[Any, ...] | None = None,
        configs_tried: int | None = None,
        last_error: Exception | None = None,
    ):
        super().__init__(message)
        self.problem_size = problem_size
        self.configs_tried = configs_tried
        self.last_error = last_error


class DeviceError(TritonKernelError):
    """Raised when tensor is not on the expected device.

    This error is raised when an input tensor is not on CUDA
    or the expected device for kernel execution.

    Attributes:
        expected_device: Expected device
        actual_device: Actual device of the tensor
        tensor_name: Name of the tensor on wrong device
    """

    def __init__(
        self,
        message: str,
        expected_device: str | None = None,
        actual_device: str | None = None,
        tensor_name: str | None = None,
    ):
        super().__init__(message)
        self.expected_device = expected_device
        self.actual_device = actual_device
        self.tensor_name = tensor_name
