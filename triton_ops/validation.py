"""Input validation utilities for Triton operators."""

from typing import List, Optional, Tuple

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
