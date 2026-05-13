"""Reference implementations for RMSNorm and RoPE.

This module provides unified reference implementations for:
- RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
- RoPE: x_rope = x * cos(theta) + rotate_half(x) * sin(theta)
- Fused RMSNorm + RoPE

Both CPU (NumPy) and GPU (PyTorch) backends are supported.

Example:
    >>> from triton_ops.reference import rmsnorm, rope, fused_rmsnorm_rope
    >>> import numpy as np
    >>>
    >>> # CPU testing
    >>> x = np.random.randn(2, 128, 4096).astype(np.float32)
    >>> weight = np.ones(4096, dtype=np.float32)
    >>> output = rmsnorm(x, weight, backend='cpu')
    >>>
    >>> # GPU verification
    >>> import torch
    >>> x_cuda = x  # Will be auto-converted to torch.Tensor
    >>> output_cuda = rmsnorm(x_cuda, weight, backend='cuda')
"""

import numpy as np
import torch

from triton_ops.reference.base import (
    Backend,
    ensure_numpy,
    ensure_torch,
    validate_backend,
)

# FP8 constants from models (single source of truth)


def rmsnorm(
    x: np.ndarray | torch.Tensor,
    weight: np.ndarray | torch.Tensor,
    eps: float = 1e-6,
    *,
    backend: Backend = "cpu",
) -> np.ndarray | torch.Tensor:
    """Reference implementation of RMSNorm.

    Formula: y = x * rsqrt(mean(x^2) + eps) * weight

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
        weight: RMSNorm weight of shape [hidden_dim]
        eps: Small constant for numerical stability (default: 1e-6)
        backend: 'cpu' for NumPy, 'cuda' for PyTorch (default: 'cpu')

    Returns:
        Normalized tensor with same shape as input

    Example:
        >>> import numpy as np
        >>> x = np.random.randn(2, 128, 4096).astype(np.float32)
        >>> weight = np.ones(4096, dtype=np.float32)
        >>> output = rmsnorm(x, weight, backend='cpu')
        >>> output.shape
        (2, 128, 4096)
    """
    validate_backend(backend)

    if backend == "cpu":
        return _rmsnorm_cpu(x, weight, eps)
    else:
        return _rmsnorm_cuda(x, weight, eps)


def _rmsnorm_cpu(
    x: np.ndarray | torch.Tensor,
    weight: np.ndarray | torch.Tensor,
    eps: float,
) -> np.ndarray:
    """NumPy implementation of RMSNorm."""
    x = ensure_numpy(x)
    weight = ensure_numpy(weight)

    # Compute mean of squares along the last dimension
    mean_sq = np.mean(x**2, axis=-1, keepdims=True)

    # Compute reciprocal RMS
    rms = np.sqrt(mean_sq + eps)
    rrms = 1.0 / rms

    # Normalize and apply weight
    output = x * rrms * weight

    return output


def _rmsnorm_cuda(
    x: np.ndarray | torch.Tensor,
    weight: np.ndarray | torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """PyTorch implementation of RMSNorm."""
    x = ensure_torch(x, device="cuda")
    weight = ensure_torch(weight, device="cuda")
    input_dtype = x.dtype

    # Compute RMS
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)

    # Normalize and apply weight
    output = (x.float() / rms * weight.float()).to(input_dtype)

    return output


def rope(
    x: np.ndarray | torch.Tensor,
    cos: np.ndarray | torch.Tensor,
    sin: np.ndarray | torch.Tensor,
    *,
    backend: Backend = "cpu",
) -> np.ndarray | torch.Tensor:
    """Reference implementation of Rotary Position Embedding (RoPE).

    Formula: x_rope = x * cos + rotate_half(x) * sin

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
        cos: Cosine embeddings of shape [seq_len, head_dim]
        sin: Sine embeddings of shape [seq_len, head_dim]
        backend: 'cpu' for NumPy, 'cuda' for PyTorch (default: 'cpu')

    Returns:
        Tensor with RoPE applied, same shape as input

    Example:
        >>> import numpy as np
        >>> x = np.random.randn(2, 128, 4096).astype(np.float32)
        >>> cos = np.random.randn(128, 64).astype(np.float32)
        >>> sin = np.random.randn(128, 64).astype(np.float32)
        >>> output = rope(x, cos, sin, backend='cpu')
        >>> output.shape
        (2, 128, 4096)
    """
    validate_backend(backend)

    if backend == "cpu":
        return _rope_cpu(x, cos, sin)
    else:
        return _rope_cuda(x, cos, sin)


def _rope_cpu(
    x: np.ndarray | torch.Tensor,
    cos: np.ndarray | torch.Tensor,
    sin: np.ndarray | torch.Tensor,
) -> np.ndarray:
    """NumPy implementation of RoPE."""
    x = ensure_numpy(x)
    cos = ensure_numpy(cos)
    sin = ensure_numpy(sin)

    head_dim = cos.shape[-1]
    hidden_dim = x.shape[-1]
    num_heads = hidden_dim // head_dim

    # Handle different input shapes
    original_shape = x.shape
    if x.ndim == 2:
        x = x[np.newaxis, :, :]
        was_2d = True
    else:
        was_2d = False

    batch, seq_len, _ = x.shape

    # Reshape for head-wise processing: [batch, seq_len, num_heads, head_dim]
    x = x.reshape(batch, seq_len, num_heads, head_dim)

    # Split into two halves for rotation
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    # Expand cos/sin for broadcasting: [1, seq_len, 1, head_dim//2]
    cos = cos[:seq_len, : head_dim // 2][np.newaxis, :, np.newaxis, :]
    sin = sin[:seq_len, : head_dim // 2][np.newaxis, :, np.newaxis, :]

    # Apply rotation
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # Concatenate and reshape back
    out = np.concatenate([out1, out2], axis=-1)

    if was_2d:
        out = out[0]

    return out.reshape(original_shape)


def _rope_cuda(
    x: np.ndarray | torch.Tensor,
    cos: np.ndarray | torch.Tensor,
    sin: np.ndarray | torch.Tensor,
) -> torch.Tensor:
    """PyTorch implementation of RoPE."""
    x = ensure_torch(x, device="cuda")
    cos = ensure_torch(cos, device="cuda")
    sin = ensure_torch(sin, device="cuda")
    input_dtype = x.dtype

    batch, seq_len, hidden_dim = x.shape
    head_dim = cos.shape[-1]
    num_heads = hidden_dim // head_dim

    # Reshape for head-wise processing
    x = x.view(batch, seq_len, num_heads, head_dim)

    # Split into two halves
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]

    # Expand cos/sin for broadcasting
    cos = cos[:seq_len, : head_dim // 2].unsqueeze(0).unsqueeze(2)
    sin = sin[:seq_len, : head_dim // 2].unsqueeze(0).unsqueeze(2)

    # Apply rotation
    out1 = x1.float() * cos.float() - x2.float() * sin.float()
    out2 = x1.float() * sin.float() + x2.float() * cos.float()

    # Concatenate and reshape back
    out = torch.cat([out1, out2], dim=-1).to(input_dtype)
    return out.view(batch, seq_len, hidden_dim)


def fused_rmsnorm_rope(
    x: np.ndarray | torch.Tensor,
    weight: np.ndarray | torch.Tensor,
    cos: np.ndarray | torch.Tensor,
    sin: np.ndarray | torch.Tensor,
    eps: float = 1e-6,
    *,
    backend: Backend = "cpu",
) -> np.ndarray | torch.Tensor:
    """Reference implementation of fused RMSNorm + RoPE.

    Applies RMSNorm first, then RoPE.

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim]
        weight: RMSNorm weight of shape [hidden_dim]
        cos: Cosine embeddings of shape [seq_len, head_dim]
        sin: Sine embeddings of shape [seq_len, head_dim]
        eps: Small constant for numerical stability (default: 1e-6)
        backend: 'cpu' for NumPy, 'cuda' for PyTorch (default: 'cpu')

    Returns:
        Output tensor with RMSNorm + RoPE applied

    Example:
        >>> import numpy as np
        >>> x = np.random.randn(2, 128, 4096).astype(np.float32)
        >>> weight = np.ones(4096, dtype=np.float32)
        >>> cos = np.random.randn(128, 64).astype(np.float32)
        >>> sin = np.random.randn(128, 64).astype(np.float32)
        >>> output = fused_rmsnorm_rope(x, weight, cos, sin, backend='cpu')
        >>> output.shape
        (2, 128, 4096)
    """
    validate_backend(backend)

    # Handle 4D cos/sin format
    if hasattr(cos, "dim") and cos.dim() == 4:
        if backend == "cuda":
            cos = cos.squeeze(0).squeeze(1)
            sin = sin.squeeze(0).squeeze(1)
        else:
            cos = ensure_numpy(cos).squeeze(0).squeeze(1)
            sin = ensure_numpy(sin).squeeze(0).squeeze(1)

    # Apply RMSNorm first
    x_norm = rmsnorm(x, weight, eps, backend=backend)

    # Then apply RoPE
    return rope(x_norm, cos, sin, backend=backend)


def compute_rope_frequencies(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    *,
    backend: Backend = "cpu",
) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
    """Compute RoPE frequency cos/sin values.

    Args:
        seq_len: Sequence length
        head_dim: Head dimension
        base: Base for frequency computation (default: 10000)
        backend: 'cpu' for NumPy, 'cuda' for PyTorch (default: 'cpu')

    Returns:
        Tuple of (cos, sin) arrays/tensors, each of shape [seq_len, head_dim]
    """
    validate_backend(backend)

    if backend == "cpu":
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))

        # Compute positions
        positions = np.arange(seq_len, dtype=np.float32)

        # Compute angles
        angles = np.outer(positions, inv_freq)

        # Compute cos and sin
        cos = np.cos(angles)
        sin = np.sin(angles)

        # Repeat to match head_dim
        cos = np.repeat(cos, 2, axis=-1)
        sin = np.repeat(sin, 2, axis=-1)

        return cos, sin
    else:
        # GPU implementation
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device="cuda") / head_dim)
        )

        positions = torch.arange(seq_len, dtype=torch.float32, device="cuda")

        angles = torch.outer(positions, inv_freq)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)

        return cos, sin
