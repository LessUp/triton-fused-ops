"""RoPE (Rotary Position Embedding) computation logic - CPU testable.

This module provides pure NumPy implementations of RoPE computation
that can be tested independently of Triton kernels.

Mathematical formula:
    x_rope = x * cos(theta) + rotate_half(x) * sin(theta)

Where rotate_half splits x into two halves and applies rotation.
"""

import numpy as np


def compute_rope(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    """Compute RoPE: x_rope = x * cos + rotate_half(x) * sin

    Rotary Position Embedding (RoPE) encodes position information through
    rotation matrices applied to pairs of features.

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
        cos: Cosine embeddings of shape [seq_len, head_dim]
        sin: Sine embeddings of shape [seq_len, head_dim]

    Returns:
        Tensor with RoPE applied, same shape as input

    Example:
        >>> x = np.random.randn(2, 128, 4096).astype(np.float32)
        >>> cos = np.random.randn(128, 64).astype(np.float32)
        >>> sin = np.random.randn(128, 64).astype(np.float32)
        >>> output = compute_rope(x, cos, sin)
        >>> output.shape
        (2, 128, 4096)

    Reference:
        Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary
        Position Embedding. arXiv preprint arXiv:2104.09864.
    """
    head_dim = cos.shape[-1]
    hidden_dim = x.shape[-1]
    num_heads = hidden_dim // head_dim

    # Handle different input shapes
    original_shape = x.shape
    if x.ndim == 2:
        x = x[np.newaxis, :, :]  # Add batch dimension
        was_2d = True
    else:
        was_2d = False

    batch, seq_len, _ = x.shape

    # Reshape for head-wise processing: [batch, seq_len, num_heads, head_dim]
    x = x.reshape(batch, seq_len, num_heads, head_dim)

    # Split into two halves for rotation: [batch, seq_len, num_heads, head_dim//2]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    # Expand cos/sin for broadcasting: [1, seq_len, 1, head_dim//2]
    cos = cos[:seq_len, : head_dim // 2][np.newaxis, :, np.newaxis, :]
    sin = sin[:seq_len, : head_dim // 2][np.newaxis, :, np.newaxis, :]

    # Apply rotation:
    # out1 = x1 * cos - x2 * sin
    # out2 = x1 * sin + x2 * cos
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # Concatenate and reshape back: [batch, seq_len, hidden_dim]
    out = np.concatenate([out1, out2], axis=-1)

    if was_2d:
        out = out[0]  # Remove batch dimension

    return out.reshape(original_shape)


def compute_rope_single_head(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> np.ndarray:
    """Compute RoPE for a single head (lower-level function).

    Args:
        x: Input of shape [head_dim]
        cos: Cosine values of shape [head_dim // 2]
        sin: Sine values of shape [head_dim // 2]

    Returns:
        Rotated vector of shape [head_dim]
    """
    head_dim = x.shape[0]
    half_dim = head_dim // 2

    x1 = x[:half_dim]
    x2 = x[half_dim:]

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return np.concatenate([out1, out2])


def compute_rope_frequencies(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
) -> tuple:
    """Compute RoPE frequency cos/sin values.

    Args:
        seq_len: Sequence length
        head_dim: Head dimension
        base: Base for frequency computation (default: 10000)

    Returns:
        Tuple of (cos, sin) arrays, each of shape [seq_len, head_dim]
    """
    # Compute inverse frequencies: theta_i = 1 / (base^(2i/d))
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

    # Compute positions: [0, 1, 2, ..., seq_len-1]
    positions = np.arange(seq_len, dtype=np.float32)

    # Compute angles: positions * inv_freq
    # Shape: [seq_len, head_dim//2]
    angles = np.outer(positions, inv_freq)

    # Compute cos and sin
    cos = np.cos(angles)
    sin = np.sin(angles)

    # Repeat to match head_dim (each freq applies to a pair)
    cos = np.repeat(cos, 2, axis=-1)
    sin = np.repeat(sin, 2, axis=-1)

    return cos, sin
