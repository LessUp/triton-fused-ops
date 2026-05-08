"""Gated MLP computation logic - CPU testable.

This module provides pure NumPy implementations of Gated MLP computation
that can be tested independently of Triton kernels.

Mathematical formula:
    output = gate_proj(x) * activation(up_proj(x))

Where activation is either SiLU (SwiGLU) or GELU (GeGLU).
"""

from typing import Literal

import numpy as np


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation function: x * sigmoid(x)

    Args:
        x: Input array

    Returns:
        Activated array
    """
    return x * (1.0 / (1.0 + np.exp(-x)))


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation function: x * 0.5 * (1 + erf(x / sqrt(2)))

    Approximation using tanh for better performance.

    Args:
        x: Input array

    Returns:
        Activated array
    """
    # Approximate GELU using tanh
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def compute_gated_mlp(
    x: np.ndarray,
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    activation: Literal["silu", "gelu"] = "silu",
) -> np.ndarray:
    """Compute Gated MLP: output = gate_proj(x) * activation(up_proj(x))

    This implements the Gated MLP used in models like LLaMA and Mistral,
    combining gate projection, up projection, and activation.

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim]
        gate_weight: Gate projection weight of shape [intermediate_dim, hidden_dim]
        up_weight: Up projection weight of shape [intermediate_dim, hidden_dim]
        activation: Activation function - "silu" (SwiGLU) or "gelu" (GeGLU)

    Returns:
        Output tensor of shape [batch, seq_len, intermediate_dim]

    Example:
        >>> x = np.random.randn(2, 128, 4096).astype(np.float32)
        >>> gate_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01
        >>> up_w = np.random.randn(11264, 4096).astype(np.float32) * 0.01
        >>> output = compute_gated_mlp(x, gate_w, up_w, activation="silu")
        >>> output.shape
        (2, 128, 11264)

    Reference:
        Shazeer, N. (2020). GLU Variants Improve Transformer.
        arXiv preprint arXiv:2002.05202.
    """
    # Reshape x for matrix multiplication: [batch * seq_len, hidden_dim]
    batch_size, seq_len, hidden_dim = x.shape
    x_flat = x.reshape(-1, hidden_dim)

    # Compute gate projection: [batch * seq_len, intermediate_dim]
    gate_proj = x_flat @ gate_weight.T

    # Compute up projection: [batch * seq_len, intermediate_dim]
    up_proj = x_flat @ up_weight.T

    # Apply activation to up projection
    if activation == "silu":
        activated = silu(up_proj)
    elif activation == "gelu":
        activated = gelu(up_proj)
    else:
        raise ValueError(f"Unknown activation: {activation}. Use 'silu' or 'gelu'.")

    # Element-wise multiply: gate * activation(up)
    output = gate_proj * activated

    # Reshape back: [batch, seq_len, intermediate_dim]
    intermediate_dim = gate_weight.shape[0]
    output = output.reshape(batch_size, seq_len, intermediate_dim)

    return output


def compute_gated_mlp_single(
    x: np.ndarray,
    gate_weight: np.ndarray,
    up_weight: np.ndarray,
    activation: Literal["silu", "gelu"] = "silu",
) -> np.ndarray:
    """Compute Gated MLP for a single position (no batch dimension).

    Args:
        x: Input of shape [hidden_dim]
        gate_weight: Gate projection weight [intermediate_dim, hidden_dim]
        up_weight: Up projection weight [intermediate_dim, hidden_dim]
        activation: Activation function

    Returns:
        Output of shape [intermediate_dim]
    """
    gate_proj = x @ gate_weight.T
    up_proj = x @ up_weight.T

    if activation == "silu":
        activated = silu(up_proj)
    elif activation == "gelu":
        activated = gelu(up_proj)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    return gate_proj * activated
