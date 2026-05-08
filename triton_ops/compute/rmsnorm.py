"""RMSNorm computation logic - CPU testable.

This module provides pure NumPy implementations of RMSNorm computation
that can be tested independently of Triton kernels.

Mathematical formula:
    y = x * rsqrt(mean(x^2) + eps) * weight
"""

import numpy as np


def compute_rmsnorm(
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight

    RMSNorm (Root Mean Square Layer Normalization) normalizes the input
    by dividing by the root mean square of the elements.

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
        weight: RMSNorm weight of shape [hidden_dim]
        eps: Small constant for numerical stability (default: 1e-6)

    Returns:
        Normalized tensor with same shape as input

    Example:
        >>> x = np.random.randn(2, 128, 4096).astype(np.float32)
        >>> weight = np.ones(4096, dtype=np.float32)
        >>> output = compute_rmsnorm(x, weight)
        >>> output.shape
        (2, 128, 4096)

    Reference:
        Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization.
        arXiv preprint arXiv:1910.07467.
    """
    # Compute mean of squares along the last dimension
    mean_sq = np.mean(x**2, axis=-1, keepdims=True)

    # Compute reciprocal RMS (root mean square)
    rms = np.sqrt(mean_sq + eps)
    rrms = 1.0 / rms

    # Normalize and apply weight
    output = x * rrms * weight

    return output


def compute_rmsnorm_row(
    x_row: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute RMSNorm for a single row.

    This is useful for verifying Triton kernel correctness on a per-row basis.

    Args:
        x_row: Input row of shape [hidden_dim]
        weight: RMSNorm weight of shape [hidden_dim]
        eps: Numerical stability constant (default: 1e-6)

    Returns:
        Normalized row of shape [hidden_dim]

    Example:
        >>> x_row = np.random.randn(4096).astype(np.float32)
        >>> weight = np.ones(4096, dtype=np.float32)
        >>> output = compute_rmsnorm_row(x_row, weight)
        >>> output.shape
        (4096,)
    """
    # Compute mean of squares
    mean_sq = np.mean(x_row**2)

    # Compute reciprocal RMS
    rrms = 1.0 / np.sqrt(mean_sq + eps)

    # Normalize and apply weight
    return x_row * rrms * weight


def compute_rms_variance(x: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the RMS variance for a tensor.

    This is useful for debugging and analyzing normalization behavior.

    Args:
        x: Input tensor
        eps: Numerical stability constant

    Returns:
        Mean squared value before normalization
    """
    return float(np.mean(x**2))
