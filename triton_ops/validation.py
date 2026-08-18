"""Input validation utilities for Triton operators."""

from typing import List, Tuple

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
    num_heads: int | None = None,
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
        if cos.shape[0] != 1 or cos.shape[1] != seq_len or cos.shape[2] != 1:
            raise ShapeMismatchError(
                "4D cos must have shape [1, seq_len, 1, head_dim]",
                expected=(1, seq_len, 1, None),
                actual=cos.shape,
                tensor_name="cos",
            )
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
    else:
        # TRIT-001/105: When num_heads is explicitly provided, validate consistency.
        if num_heads * head_dim != hidden_dim:
            raise ShapeMismatchError(
                f"num_heads ({num_heads}) * head_dim ({head_dim}) = {num_heads * head_dim} "
                f"!= hidden_dim ({hidden_dim})",
                tensor_name="x",
            )

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


def validate_flash_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[int, int, int, int]:
    """Validate Q, K and V before launching the FlashAttention kernel."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ShapeMismatchError(
            f"q, k and v must be 4D; got {q.dim()}D, {k.dim()}D and {v.dim()}D"
        )
    if q.shape != k.shape or q.shape != v.shape:
        raise ShapeMismatchError(
            f"q, k and v shapes must match; got {q.shape}, {k.shape} and {v.shape}"
        )

    _check_same_device((q, "q"), (k, "k"), (v, "v"))
    _check_dtype(q, "q", SUPPORTED_DTYPES_FLOAT)
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise UnsupportedDtypeError(
            f"q, k and v dtypes must match; got {q.dtype}, {k.dtype} and {v.dtype}"
        )

    _check_cuda(q, "q")
    _check_cuda(k, "k")
    _check_cuda(v, "v")

    batch, heads, seq_len, head_dim = q.shape
    validate_positive_dimensions(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        head_dim=head_dim,
    )
    if head_dim not in (16, 32, 64, 128):
        raise ValueError(f"head_dim must be one of 16, 32, 64 or 128; got {head_dim}")

    return batch, heads, seq_len, head_dim


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
