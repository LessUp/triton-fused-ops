"""Input validation utilities for Triton operators.

Design Philosophy
-----------------
This module provides both procedural validation functions and a declarative
validator framework. The declarative approach allows kernel functions to
specify their input contracts concisely, while the procedural helpers remain
available for backward compatibility and fine-grained control.

Key concepts:
- **InputContract**: Declarative specification of kernel input requirements
- **validate_contract()**: One-line validation for kernel functions
- **validate_*() functions**: Traditional procedural validators (preserved for compatibility)

For CPU testing without GPU, use `triton_ops.reference` module instead.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from triton_ops.exceptions import (
    DeviceError,
    ShapeMismatchError,
    UnsupportedDtypeError,
)

# Activation type constants (moved from utils.py)
ACTIVATION_SILU = "silu"
ACTIVATION_GELU = "gelu"
VALID_ACTIVATIONS = (ACTIVATION_SILU, ACTIVATION_GELU)

# Supported dtypes for different operations
SUPPORTED_DTYPES_FLOAT = [torch.float16, torch.bfloat16, torch.float32]
SUPPORTED_DTYPES_FP8 = [torch.uint8]  # FP8 represented as uint8 when native FP8 not available

# Add native FP8 types if available (PyTorch 2.1+)
if hasattr(torch, "float8_e4m3fn"):
    SUPPORTED_DTYPES_FP8.append(torch.float8_e4m3fn)
if hasattr(torch, "float8_e5m2"):
    SUPPORTED_DTYPES_FP8.append(torch.float8_e5m2)


def _check_cuda(tensor: torch.Tensor, tensor_name: str) -> None:
    """Check that tensor is on CUDA device.

    Args:
        tensor: Tensor to check
        tensor_name: Name for error messages

    Raises:
        DeviceError: If tensor is not on CUDA
    """
    if not tensor.is_cuda:
        raise DeviceError(
            f"{tensor_name} must be on CUDA device, got {tensor.device}",
            expected_device="cuda",
            actual_device=str(tensor.device),
            tensor_name=tensor_name,
        )


def _check_dtype(
    tensor: torch.Tensor,
    tensor_name: str,
    supported_dtypes: List[torch.dtype],
) -> None:
    """Check that tensor has a supported dtype.

    Args:
        tensor: Tensor to check
        tensor_name: Name for error messages
        supported_dtypes: List of supported dtypes

    Raises:
        UnsupportedDtypeError: If tensor dtype is not supported
    """
    if tensor.dtype not in supported_dtypes:
        raise UnsupportedDtypeError(
            f"{tensor_name} has unsupported dtype {tensor.dtype}, supported: {supported_dtypes}",
            dtype=tensor.dtype,
            supported_dtypes=supported_dtypes,
            tensor_name=tensor_name,
        )


def _check_contiguous(tensor: torch.Tensor, tensor_name: str) -> None:
    """Check that tensor is contiguous.

    Args:
        tensor: Tensor to check
        tensor_name: Name for error messages

    Raises:
        ValueError: If tensor is not contiguous
    """
    if not tensor.is_contiguous():
        raise ValueError(f"{tensor_name} must be contiguous")


def _check_same_device(*tensors: Tuple[torch.Tensor, str]) -> None:
    """Check that all tensors are on the same device.

    Args:
        *tensors: Tuples of (tensor, tensor_name)

    Raises:
        DeviceError: If tensors are on different devices
    """
    if len(tensors) < 2:
        return

    first_device = tensors[0][0].device
    for tensor, name in tensors[1:]:
        if tensor.device != first_device:
            raise DeviceError(
                f"{name} is on {tensor.device} but expected {first_device}",
                expected_device=str(first_device),
                actual_device=str(tensor.device),
                tensor_name=name,
            )


def validate_rmsnorm_rope_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: Optional[int] = None,
) -> Tuple[int, int, int, int, int]:
    """Validate inputs for RMSNorm + RoPE kernel.

    Args:
        x: Input tensor [batch, seq_len, hidden_dim]
        weight: RMSNorm weight [hidden_dim]
        cos: Cosine embeddings [seq_len, head_dim] or [1, seq_len, 1, head_dim]
        sin: Sine embeddings [seq_len, head_dim] or [1, seq_len, 1, head_dim]
        num_heads: Optional number of attention heads

    Returns:
        Tuple of (batch_size, seq_len, hidden_dim, head_dim, num_heads)

    Raises:
        ShapeMismatchError: If tensor shapes are incompatible
        UnsupportedDtypeError: If tensor dtypes are unsupported
        DeviceError: If tensors are not on CUDA device
    """
    # Check CUDA
    _check_cuda(x, "x")
    _check_cuda(weight, "weight")
    _check_cuda(cos, "cos")
    _check_cuda(sin, "sin")

    # Check all tensors on same device
    _check_same_device((x, "x"), (weight, "weight"), (cos, "cos"), (sin, "sin"))

    # Check dtypes
    _check_dtype(x, "x", SUPPORTED_DTYPES_FLOAT)
    _check_dtype(weight, "weight", SUPPORTED_DTYPES_FLOAT)
    _check_dtype(cos, "cos", SUPPORTED_DTYPES_FLOAT)
    _check_dtype(sin, "sin", SUPPORTED_DTYPES_FLOAT)

    # Check contiguous
    _check_contiguous(x, "x")
    _check_contiguous(weight, "weight")
    _check_contiguous(cos, "cos")
    _check_contiguous(sin, "sin")

    # Check x shape
    if x.dim() != 3:
        raise ShapeMismatchError(
            f"x must be 3D [batch, seq_len, hidden_dim], got {x.dim()}D",
            expected=(None, None, None),
            actual=x.shape,
            tensor_name="x",
        )

    batch_size, seq_len, hidden_dim = x.shape

    # Check weight shape
    if weight.shape != (hidden_dim,):
        raise ShapeMismatchError(
            f"weight shape {weight.shape} doesn't match hidden_dim {hidden_dim}",
            expected=(hidden_dim,),
            actual=weight.shape,
            tensor_name="weight",
        )

    # Determine head_dim from cos/sin
    if cos.dim() == 2:
        # [seq_len, head_dim]
        if cos.shape[0] != seq_len:
            raise ShapeMismatchError(
                f"cos seq_len {cos.shape[0]} doesn't match x seq_len {seq_len}",
                expected=(seq_len, None),
                actual=cos.shape,
                tensor_name="cos",
            )
        head_dim = cos.shape[1]
    elif cos.dim() == 4:
        # [1, seq_len, 1, head_dim]
        head_dim = cos.shape[3]
    else:
        raise ShapeMismatchError(
            f"cos must be 2D or 4D, got {cos.dim()}D",
            tensor_name="cos",
        )

    # Check sin matches cos
    if sin.shape != cos.shape:
        raise ShapeMismatchError(
            f"sin shape {sin.shape} doesn't match cos shape {cos.shape}",
            expected=cos.shape,
            actual=sin.shape,
            tensor_name="sin",
        )

    # Compute num_heads if not provided
    if num_heads is None:
        if hidden_dim % head_dim != 0:
            raise ShapeMismatchError(
                f"hidden_dim {hidden_dim} must be divisible by head_dim {head_dim}",
                tensor_name="x",
            )
        num_heads = hidden_dim // head_dim

    return batch_size, seq_len, hidden_dim, head_dim, num_heads


def validate_gated_mlp_inputs(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    activation: str = "silu",
) -> Tuple[int, int, int, int]:
    """Validate inputs for Gated MLP kernel.

    Args:
        x: Input tensor [batch, seq_len, hidden_dim]
        gate_weight: Gate projection weight [intermediate_dim, hidden_dim]
        up_weight: Up projection weight [intermediate_dim, hidden_dim]
        activation: Activation function ("silu" or "gelu")

    Returns:
        Tuple of (batch_size, seq_len, hidden_dim, intermediate_dim)

    Raises:
        ShapeMismatchError: If tensor shapes are incompatible
        UnsupportedDtypeError: If tensor dtypes are unsupported
        DeviceError: If tensors are not on CUDA device
        ValueError: If activation is not supported
    """
    # Check CUDA
    _check_cuda(x, "x")
    _check_cuda(gate_weight, "gate_weight")
    _check_cuda(up_weight, "up_weight")

    # Check all tensors on same device
    _check_same_device((x, "x"), (gate_weight, "gate_weight"), (up_weight, "up_weight"))

    # Check dtypes
    _check_dtype(x, "x", SUPPORTED_DTYPES_FLOAT)
    _check_dtype(gate_weight, "gate_weight", SUPPORTED_DTYPES_FLOAT)
    _check_dtype(up_weight, "up_weight", SUPPORTED_DTYPES_FLOAT)

    # Check contiguous
    _check_contiguous(x, "x")
    _check_contiguous(gate_weight, "gate_weight")
    _check_contiguous(up_weight, "up_weight")

    # Check activation
    if activation not in VALID_ACTIVATIONS:
        raise ValueError(f"activation must be 'silu' or 'gelu', got '{activation}'")

    # Check x shape
    if x.dim() != 3:
        raise ShapeMismatchError(
            f"x must be 3D [batch, seq_len, hidden_dim], got {x.dim()}D",
            tensor_name="x",
        )

    batch_size, seq_len, hidden_dim = x.shape

    # Check gate_weight shape
    if gate_weight.dim() != 2:
        raise ShapeMismatchError(
            f"gate_weight must be 2D [intermediate_dim, hidden_dim], got {gate_weight.dim()}D",
            tensor_name="gate_weight",
        )

    intermediate_dim, gate_hidden = gate_weight.shape
    if gate_hidden != hidden_dim:
        raise ShapeMismatchError(
            f"gate_weight hidden_dim {gate_hidden} doesn't match x hidden_dim {hidden_dim}",
            expected=(None, hidden_dim),
            actual=gate_weight.shape,
            tensor_name="gate_weight",
        )

    # Check up_weight shape
    if up_weight.shape != gate_weight.shape:
        raise ShapeMismatchError(
            f"up_weight shape {up_weight.shape} doesn't match gate_weight shape {gate_weight.shape}",
            expected=gate_weight.shape,
            actual=up_weight.shape,
            tensor_name="up_weight",
        )

    return batch_size, seq_len, hidden_dim, intermediate_dim


def validate_fp8_gemm_inputs(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: Optional[torch.Tensor],
    b_scale: Optional[torch.Tensor],
    output_dtype: torch.dtype = torch.float16,
) -> Tuple[int, int, int]:
    """Validate inputs for FP8 GEMM kernel.

    Args:
        a: First matrix [M, K] in FP8 or float
        b: Second matrix [K, N] in FP8 or float
        a_scale: Scaling factor for A (required if A is FP8)
        b_scale: Scaling factor for B (required if B is FP8)
        output_dtype: Output data type

    Returns:
        Tuple of (M, N, K)

    Raises:
        ShapeMismatchError: If tensor shapes are incompatible
        UnsupportedDtypeError: If tensor dtypes are unsupported
        DeviceError: If tensors are not on CUDA device
        ValueError: If FP8 inputs are missing required scale factors
    """
    # Check CUDA
    _check_cuda(a, "a")
    _check_cuda(b, "b")

    # Check FP8 scale requirements before other validations
    if a.dtype in SUPPORTED_DTYPES_FP8 and a_scale is None:
        raise ValueError(
            f"a_scale is required when A is FP8 (dtype={a.dtype}). "
            "Provide the scale factor used during quantization."
        )
    if b.dtype in SUPPORTED_DTYPES_FP8 and b_scale is None:
        raise ValueError(
            f"b_scale is required when B is FP8 (dtype={b.dtype}). "
            "Provide the scale factor used during quantization."
        )

    # Check scale tensors if provided
    if a_scale is not None:
        _check_cuda(a_scale, "a_scale")
    if b_scale is not None:
        _check_cuda(b_scale, "b_scale")

    # Check all tensors on same device
    tensors_to_check = [(a, "a"), (b, "b")]
    if a_scale is not None:
        tensors_to_check.append((a_scale, "a_scale"))
    if b_scale is not None:
        tensors_to_check.append((b_scale, "b_scale"))
    _check_same_device(*tensors_to_check)

    # Check contiguous
    _check_contiguous(a, "a")
    _check_contiguous(b, "b")

    # Check output dtype
    if output_dtype not in [torch.float16, torch.bfloat16, torch.float32]:
        raise UnsupportedDtypeError(
            f"output_dtype must be float16, bfloat16, or float32, got {output_dtype}",
            dtype=output_dtype,
            supported_dtypes=[torch.float16, torch.bfloat16, torch.float32],
        )

    # Check matrix shapes
    if a.dim() != 2:
        raise ShapeMismatchError(
            f"a must be 2D [M, K], got {a.dim()}D",
            tensor_name="a",
        )

    if b.dim() != 2:
        raise ShapeMismatchError(
            f"b must be 2D [K, N], got {b.dim()}D",
            tensor_name="b",
        )

    M, K_a = a.shape
    K_b, N = b.shape

    if K_a != K_b:
        raise ShapeMismatchError(
            f"Matrix dimensions don't match: a is [{M}, {K_a}], b is [{K_b}, {N}]",
            tensor_name="a, b",
        )

    return M, N, K_a


def validate_fp8_quantize_inputs(
    tensor: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
) -> None:
    """Validate inputs for FP8 quantization.

    Args:
        tensor: Input tensor to quantize
        scale: Optional pre-computed scale factor

    Raises:
        UnsupportedDtypeError: If tensor dtype is not supported
        DeviceError: If tensor is not on CUDA device
    """
    _check_cuda(tensor, "tensor")
    _check_dtype(tensor, "tensor", SUPPORTED_DTYPES_FLOAT)
    _check_contiguous(tensor, "tensor")

    if scale is not None:
        _check_cuda(scale, "scale")
        _check_dtype(scale, "scale", [torch.float32])
        if scale.numel() != 1:
            raise ShapeMismatchError(
                f"scale must be a scalar tensor, got shape {scale.shape}",
                expected=(),
                actual=scale.shape,
                tensor_name="scale",
            )
        if scale.item() <= 0:
            raise ValueError(f"scale must be positive, got {scale.item()}")


def validate_positive_dimensions(**dims: int) -> None:
    """Validate that all dimensions are positive.

    Args:
        **dims: Dimension name-value pairs

    Raises:
        ValueError: If any dimension is not positive
    """
    for name, value in dims.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def validate_head_dim(head_dim: int) -> None:
    """Validate that head_dim is even (required for RoPE rotation).

    Args:
        head_dim: Head dimension size

    Raises:
        ValueError: If head_dim is not even
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE rotation, got {head_dim}")


def validate_eps(eps: float) -> None:
    """Validate epsilon value for numerical stability.

    Args:
        eps: Epsilon value

    Raises:
        ValueError: If eps is not positive
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")


# =============================================================================
# Declarative Validation Framework
# =============================================================================


@dataclass
class TensorContract:
    """Declarative specification for a single tensor's requirements.

    Attributes:
        name: Name of the tensor for error messages
        ndim: Expected number of dimensions (None = any)
        dtype: Expected dtype or list of supported dtypes (None = any float)
        device: Expected device ("cuda" or None for any)
        contiguous: Whether tensor must be contiguous
        shape: Expected shape as tuple (use None for wildcard dimensions)
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
    """

    name: str
    ndim: Optional[int] = None
    dtype: Optional[Union[torch.dtype, List[torch.dtype]]] = None
    device: Optional[str] = "cuda"
    contiguous: bool = True
    shape: Optional[Tuple[Optional[int], ...]] = None
    min_dims: Optional[int] = None
    max_dims: Optional[int] = None

    def validate(self, tensor: torch.Tensor) -> Tuple[bool, Optional[str]]:
        """Validate a tensor against this contract.

        Args:
            tensor: Tensor to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check device
        if self.device == "cuda" and not tensor.is_cuda:
            return False, f"{self.name} must be on CUDA device, got {tensor.device}"

        # Check contiguous
        if self.contiguous and not tensor.is_contiguous():
            return False, f"{self.name} must be contiguous"

        # Check dtype
        if self.dtype is not None:
            allowed_dtypes = self.dtype if isinstance(self.dtype, list) else [self.dtype]
            if tensor.dtype not in allowed_dtypes:
                return False, f"{self.name} has unsupported dtype {tensor.dtype}"

        # Check ndim
        if self.ndim is not None and tensor.dim() != self.ndim:
            return False, f"{self.name} must be {self.ndim}D, got {tensor.dim()}D"

        # Check min/max dims
        if self.min_dims is not None and tensor.dim() < self.min_dims:
            return False, f"{self.name} must have at least {self.min_dims} dims, got {tensor.dim()}"
        if self.max_dims is not None and tensor.dim() > self.max_dims:
            return False, f"{self.name} must have at most {self.max_dims} dims, got {tensor.dim()}"

        # Check shape pattern
        if self.shape is not None:
            if len(self.shape) != tensor.dim():
                return (
                    False,
                    f"{self.name} shape {tensor.shape} doesn't match expected pattern {self.shape}",
                )
            for i, (expected, actual) in enumerate(zip(self.shape, tensor.shape)):
                if expected is not None and expected != actual:
                    return False, f"{self.name} dimension {i} must be {expected}, got {actual}"

        return True, None


@dataclass
class InputContract:
    """Declarative specification for all inputs to a kernel function.

    This allows kernel functions to specify their input requirements concisely
    and have all validation performed in one call.

    Example:
        >>> contract = InputContract(
        ...     tensors=[
        ...         TensorContract("x", ndim=3, dtype=[torch.float16, torch.bfloat16]),
        ...         TensorContract("weight", ndim=1, shape=(None,)),  # last dim inferred
        ...     ],
        ...     scalar_params={"eps": lambda v: v > 0},
        ...     same_device=True,
        ... )
        >>> result = contract.validate(x, weight, eps=1e-6)
        >>> if result.is_valid:
        ...     batch, seq, hidden = result.dims
    """

    tensors: List[TensorContract] = field(default_factory=list)
    scalar_params: Dict[str, Callable[[Any], bool]] = field(default_factory=dict)
    same_device: bool = True
    same_dtype: bool = False
    shape_relations: Dict[str, str] = field(default_factory=dict)  # e.g., {"x:hidden": "weight:0"}

    def validate(self, *tensors: torch.Tensor, **scalars: Any) -> "ContractResult":
        """Validate all inputs against this contract.

        Args:
            *tensors: Tensor inputs in order matching self.tensors
            **scalars: Scalar parameters

        Returns:
            ContractResult with validation status and extracted dimensions
        """
        errors: List[str] = []

        # Validate tensor count
        if len(tensors) != len(self.tensors):
            raise ValueError(f"Expected {len(self.tensors)} tensors, got {len(tensors)}")

        # Validate each tensor
        validated_tensors = {}
        for contract, tensor in zip(self.tensors, tensors):
            is_valid, error = contract.validate(tensor)
            if not is_valid and error is not None:
                errors.append(error)
            validated_tensors[contract.name] = tensor

        # Validate scalar parameters
        for param_name, validator in self.scalar_params.items():
            if param_name in scalars:
                value = scalars[param_name]
                if not validator(value):
                    errors.append(f"Invalid value for {param_name}: {value}")

        # Check same device constraint
        if self.same_device and len(tensors) > 1:
            devices = set(t.device for t in tensors)
            if len(devices) > 1:
                errors.append(f"All tensors must be on same device, got {devices}")

        # Check same dtype constraint
        if self.same_dtype and len(tensors) > 1:
            dtypes = set(t.dtype for t in tensors)
            if len(dtypes) > 1:
                errors.append(f"All tensors must have same dtype, got {dtypes}")

        return ContractResult(
            is_valid=len(errors) == 0,
            errors=errors,
            tensors=validated_tensors,
        )


@dataclass
class ContractResult:
    """Result of validating inputs against a contract."""

    is_valid: bool
    errors: List[str]
    tensors: Dict[str, torch.Tensor]

    @property
    def dims(self) -> Tuple[int, ...]:
        """Extract dimensions from validated tensors.

        Returns tuple of all dimension sizes in order.
        """
        result = []
        for tensor in self.tensors.values():
            result.extend(tensor.shape)
        return tuple(result)

    def raise_if_invalid(self) -> None:
        """Raise appropriate exception if validation failed."""
        if not self.is_valid:
            error_msg = "; ".join(self.errors)
            # Determine most appropriate exception type
            if "CUDA" in error_msg or "device" in error_msg.lower():
                raise DeviceError(error_msg)
            elif "shape" in error_msg.lower() or "dimension" in error_msg.lower():
                raise ShapeMismatchError(error_msg)
            elif "dtype" in error_msg.lower():
                raise UnsupportedDtypeError(error_msg)
            else:
                raise ValueError(error_msg)


# =============================================================================
# Pre-defined contracts for each kernel family
# =============================================================================

RMSNORM_ROPE_CONTRACT = InputContract(
    tensors=[
        TensorContract("x", ndim=3, dtype=SUPPORTED_DTYPES_FLOAT),
        TensorContract("weight", ndim=1, dtype=SUPPORTED_DTYPES_FLOAT),
        TensorContract("cos", min_dims=2, max_dims=4, dtype=SUPPORTED_DTYPES_FLOAT),
        TensorContract("sin", min_dims=2, max_dims=4, dtype=SUPPORTED_DTYPES_FLOAT),
    ],
    scalar_params={
        "eps": lambda v: isinstance(v, (int, float)) and v > 0,
        "num_heads": lambda v: v is None or (isinstance(v, int) and v > 0),
    },
    same_device=True,
    same_dtype=True,
)

GATED_MLP_CONTRACT = InputContract(
    tensors=[
        TensorContract("x", ndim=3, dtype=SUPPORTED_DTYPES_FLOAT),
        TensorContract("gate_weight", ndim=2, dtype=SUPPORTED_DTYPES_FLOAT),
        TensorContract("up_weight", ndim=2, dtype=SUPPORTED_DTYPES_FLOAT),
    ],
    scalar_params={
        "activation": lambda v: v in VALID_ACTIVATIONS,
    },
    same_device=True,
    same_dtype=True,
)

FP8_GEMM_CONTRACT = InputContract(
    tensors=[
        TensorContract("a", ndim=2, dtype=None),  # Accepts FP8 or float
        TensorContract("b", ndim=2, dtype=None),
        TensorContract("a_scale", ndim=0, dtype=[torch.float32], contiguous=False),
        TensorContract("b_scale", ndim=0, dtype=[torch.float32], contiguous=False),
    ],
    scalar_params={
        "output_dtype": lambda v: v in [torch.float16, torch.bfloat16, torch.float32],
    },
    same_device=True,
)

FP8_QUANTIZE_CONTRACT = InputContract(
    tensors=[
        TensorContract("tensor", min_dims=1, dtype=SUPPORTED_DTYPES_FLOAT),
        TensorContract("scale", ndim=0, dtype=[torch.float32], contiguous=False),
    ],
    scalar_params={},
    same_device=True,
)


def validate_with_contract(
    contract: InputContract,
    *tensors: torch.Tensor,
    **scalars: Any,
) -> ContractResult:
    """Validate inputs against a contract and return result.

    This is a convenience function for one-line validation.

    Args:
        contract: The input contract to validate against
        *tensors: Tensor inputs
        **scalars: Scalar parameters

    Returns:
        ContractResult with validation status

    Example:
        >>> result = validate_with_contract(RMSNORM_ROPE_CONTRACT, x, weight, cos, sin, eps=1e-6)
        >>> result.raise_if_invalid()
    """
    return contract.validate(*tensors, **scalars)
