"""Reference implementations for Gated MLP.

This module provides unified reference implementations for Gated MLP
used in models like LLaMA and Mistral.

Formula: output = gate_proj(x) * activation(up_proj(x))

Where activation is either SiLU (SwiGLU) or GELU (GeGLU).

Both CPU (NumPy) and GPU (PyTorch) backends are supported.

Example:
    >>> from triton_ops.reference import gated_mlp
    >>> import numpy as np
    >>>
    >>> # CPU testing
    >>> x = np.random.randn(2, 128, 4096).astype(np.float32)
    >>> gate_w = np.random.randn(11008, 4096).astype(np.float32) * 0.01
    >>> up_w = np.random.randn(11008, 4096).astype(np.float32) * 0.01
    >>> output = gated_mlp(x, gate_w, up_w, activation='silu', backend='cpu')
    >>> output.shape
    (2, 128, 11008)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from triton_ops.reference.base import (
    Backend,
    ensure_numpy,
    ensure_torch,
    validate_backend,
)

Activation = Literal["silu", "gelu"]


def gated_mlp(
    x: np.ndarray | torch.Tensor,
    gate_weight: np.ndarray | torch.Tensor,
    up_weight: np.ndarray | torch.Tensor,
    activation: Activation = "silu",
    *,
    backend: Backend = "cpu",
) -> np.ndarray | torch.Tensor:
    """Reference implementation of Gated MLP.

    Formula: output = gate_proj(x) * activation(up_proj(x))

    This implements the Gated MLP used in models like LLaMA and Mistral,
    combining gate projection, up projection, and activation.

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim]
        gate_weight: Gate projection weight of shape [intermediate_dim, hidden_dim]
        up_weight: Up projection weight of shape [intermediate_dim, hidden_dim]
        activation: Activation function - "silu" (SwiGLU) or "gelu" (GeGLU)
        backend: 'cpu' for NumPy, 'cuda' for PyTorch (default: 'cpu')

    Returns:
        Output tensor of shape [batch, seq_len, intermediate_dim]

    Raises:
        ValueError: If activation is not 'silu' or 'gelu'

    Example:
        >>> import numpy as np
        >>> x = np.random.randn(2, 128, 4096).astype(np.float32)
        >>> gate_w = np.random.randn(11008, 4096).astype(np.float32) * 0.01
        >>> up_w = np.random.randn(11008, 4096).astype(np.float32) * 0.01
        >>> output = gated_mlp(x, gate_w, up_w, activation='silu', backend='cpu')
        >>> output.shape
        (2, 128, 11008)
    """
    validate_backend(backend)

    if activation not in ("silu", "gelu"):
        raise ValueError(f"activation must be 'silu' or 'gelu', got {activation!r}")

    if backend == "cpu":
        return _gated_mlp_cpu(x, gate_weight, up_weight, activation)
    else:
        return _gated_mlp_cuda(x, gate_weight, up_weight, activation)


def _silu_cpu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation: x * sigmoid(x)"""
    return x * (1.0 / (1.0 + np.exp(-x)))


def _gelu_cpu(x: np.ndarray) -> np.ndarray:
    """GELU activation using tanh approximation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def _gated_mlp_cpu(
    x: np.ndarray | torch.Tensor,
    gate_weight: np.ndarray | torch.Tensor,
    up_weight: np.ndarray | torch.Tensor,
    activation: Activation,
) -> np.ndarray:
    """NumPy implementation of Gated MLP."""
    x = ensure_numpy(x)
    gate_weight = ensure_numpy(gate_weight)
    up_weight = ensure_numpy(up_weight)

    # Reshape x for matrix multiplication: [batch * seq_len, hidden_dim]
    batch_size, seq_len, hidden_dim = x.shape
    x_flat = x.reshape(-1, hidden_dim)

    # Compute gate projection: [batch * seq_len, intermediate_dim]
    gate_proj = x_flat @ gate_weight.T

    # Compute up projection: [batch * seq_len, intermediate_dim]
    up_proj = x_flat @ up_weight.T

    # Apply activation to up projection (standard SwiGLU formula)
    if activation == "silu":
        activated = _silu_cpu(up_proj)
    else:
        activated = _gelu_cpu(up_proj)

    # Element-wise multiply: gate * activation(up)
    output = gate_proj * activated

    # Reshape back: [batch, seq_len, intermediate_dim]
    intermediate_dim = gate_weight.shape[0]
    output = output.reshape(batch_size, seq_len, intermediate_dim)

    return output


def _gated_mlp_cuda(
    x: np.ndarray | torch.Tensor,
    gate_weight: np.ndarray | torch.Tensor,
    up_weight: np.ndarray | torch.Tensor,
    activation: Activation,
) -> torch.Tensor:
    """PyTorch implementation of Gated MLP."""
    x = ensure_torch(x, device="cuda")
    gate_weight = ensure_torch(gate_weight, device="cuda")
    up_weight = ensure_torch(up_weight, device="cuda")
    input_dtype = x.dtype

    # Compute projections
    gate = torch.nn.functional.linear(x.float(), gate_weight.float())
    up = torch.nn.functional.linear(x.float(), up_weight.float())

    # Apply activation to gate projection (standard SwiGLU)
    if activation == "silu":
        gate_activated = torch.nn.functional.silu(gate)
    else:
        gate_activated = torch.nn.functional.gelu(gate)

    # Gated output: activation(gate_proj(x)) * up_proj(x)
    output = gate_activated * up

    return output.to(input_dtype)
