"""Fused RMSNorm + RoPE Triton kernel implementation.

This module implements a fused kernel that combines RMSNorm and Rotary Position Embedding (RoPE)
into a single GPU kernel, reducing HBM access from 3 to 1.

Mathematical formulas:
- RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
- RoPE: x_rope = x * cos(theta) + rotate_half(x) * sin(theta)
"""

import torch
import triton
import triton.language as tl

from triton_ops.validation import validate_rmsnorm_rope_inputs


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    output_ptr,
    weight_ptr,
    stride_x_batch,
    stride_x_seq,
    stride_x_hidden,
    stride_out_batch,
    stride_out_seq,
    stride_out_hidden,
    hidden_dim,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """RMSNorm kernel: y = x * rsqrt(mean(x^2) + eps) * weight

    Each program instance processes one row (one position in one batch).
    """
    # Get row index
    row_idx = tl.program_id(0)
    batch_idx = row_idx // tl.cdiv(hidden_dim, BLOCK_SIZE)

    # Compute row start pointer
    row_start = x_ptr + row_idx * stride_x_seq
    out_start = output_ptr + row_idx * stride_out_seq

    # Load row and compute sum of squares
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_dim
        x = tl.load(row_start + cols * stride_x_hidden, mask=mask, other=0.0)
        sum_sq += tl.where(mask, x.to(tl.float32) * x.to(tl.float32), 0.0)

    # Compute RMS
    mean_sq = tl.sum(sum_sq) / hidden_dim
    rrms = tl.rsqrt(mean_sq + eps)

    # Normalize and apply weight
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_dim
        x = tl.load(row_start + cols * stride_x_hidden, mask=mask, other=0.0)
        w = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        out = x.to(tl.float32) * rrms * w.to(tl.float32)
        tl.store(out_start + cols * stride_out_hidden, out.to(x.dtype), mask=mask)


@triton.jit
def rope_kernel(
    x_ptr,
    output_ptr,
    cos_ptr,
    sin_ptr,
    stride_x_batch,
    stride_x_seq,
    stride_x_hidden,
    stride_out_batch,
    stride_out_seq,
    stride_out_hidden,
    stride_cos_seq,
    stride_cos_dim,
    seq_len,
    hidden_dim,
    head_dim,
    num_heads,
    BLOCK_SIZE: tl.constexpr,
):
    """RoPE kernel: x_rope = x * cos + rotate_half(x) * sin

    Applies rotary position embedding to each head independently.
    """
    # Get position in grid
    pid = tl.program_id(0)
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len

    # Base pointers for this position
    x_base = x_ptr + batch_idx * stride_x_batch + seq_idx * stride_x_seq
    out_base = output_ptr + batch_idx * stride_out_batch + seq_idx * stride_out_seq
    cos_base = cos_ptr + seq_idx * stride_cos_seq
    sin_base = sin_ptr + seq_idx * stride_cos_seq

    # Process each head
    half_head = head_dim // 2

    for head_idx in range(num_heads):
        head_offset = head_idx * head_dim

        # Process pairs of elements for rotation
        for i in range(0, half_head, BLOCK_SIZE):
            cols = i + tl.arange(0, BLOCK_SIZE)
            mask = cols < half_head

            # Load x values (first half and second half of head)
            x1_idx = head_offset + cols
            x2_idx = head_offset + half_head + cols

            x1 = tl.load(x_base + x1_idx * stride_x_hidden, mask=mask, other=0.0)
            x2 = tl.load(x_base + x2_idx * stride_x_hidden, mask=mask, other=0.0)

            # Load cos and sin
            cos_val = tl.load(cos_base + cols * stride_cos_dim, mask=mask, other=1.0)
            sin_val = tl.load(sin_base + cols * stride_cos_dim, mask=mask, other=0.0)

            # Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
            out1 = x1.to(tl.float32) * cos_val.to(tl.float32) - x2.to(tl.float32) * sin_val.to(
                tl.float32
            )
            out2 = x1.to(tl.float32) * sin_val.to(tl.float32) + x2.to(tl.float32) * cos_val.to(
                tl.float32
            )

            # Store results
            tl.store(out_base + x1_idx * stride_out_hidden, out1.to(x1.dtype), mask=mask)
            tl.store(out_base + x2_idx * stride_out_hidden, out2.to(x1.dtype), mask=mask)


@triton.jit
def fused_rmsnorm_rope_kernel(
    x_ptr,
    output_ptr,
    weight_ptr,
    cos_ptr,
    sin_ptr,
    stride_x_batch,
    stride_x_seq,
    stride_x_hidden,
    stride_out_batch,
    stride_out_seq,
    stride_out_hidden,
    stride_cos_seq,
    stride_cos_dim,
    batch_size,
    seq_len,
    hidden_dim,
    head_dim,
    num_heads,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused RMSNorm + RoPE kernel.

    Combines RMSNorm and RoPE into a single kernel to minimize HBM access.
    Each program instance processes one row (one position in one batch).

    Memory access pattern:
    - Without fusion: HBM -> RMSNorm -> HBM -> RoPE -> HBM (3 HBM accesses)
    - With fusion: HBM -> [RMSNorm + RoPE in registers] -> HBM (1 HBM access)
    """
    # Get row index (batch * seq_len + seq_idx)
    row_idx = tl.program_id(0)
    batch_idx = row_idx // seq_len
    seq_idx = row_idx % seq_len

    # Skip if out of bounds
    if batch_idx >= batch_size:
        return

    # Compute base pointers
    x_row = x_ptr + batch_idx * stride_x_batch + seq_idx * stride_x_seq
    out_row = output_ptr + batch_idx * stride_out_batch + seq_idx * stride_out_seq
    cos_row = cos_ptr + seq_idx * stride_cos_seq
    sin_row = sin_ptr + seq_idx * stride_cos_seq

    # Step 1: Compute RMS (sum of squares)
    sum_sq = tl.zeros([1], dtype=tl.float32)

    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < hidden_dim
        x = tl.load(x_row + cols * stride_x_hidden, mask=mask, other=0.0)
        sum_sq += tl.sum(tl.where(mask, x.to(tl.float32) * x.to(tl.float32), 0.0))

    # Compute inverse RMS
    mean_sq = sum_sq / hidden_dim
    rrms = tl.rsqrt(mean_sq + eps)

    # Step 2: Apply RMSNorm and RoPE together
    half_head = head_dim // 2

    for head_idx in range(num_heads):
        head_offset = head_idx * head_dim

        # Process pairs for RoPE rotation
        for i in range(0, half_head, BLOCK_SIZE):
            cols = i + tl.arange(0, BLOCK_SIZE)
            mask = cols < half_head

            # Indices for first and second half of head
            idx1 = head_offset + cols
            idx2 = head_offset + half_head + cols

            # Load x values
            x1 = tl.load(x_row + idx1 * stride_x_hidden, mask=mask, other=0.0)
            x2 = tl.load(x_row + idx2 * stride_x_hidden, mask=mask, other=0.0)

            # Load weights
            w1 = tl.load(weight_ptr + idx1, mask=mask, other=0.0)
            w2 = tl.load(weight_ptr + idx2, mask=mask, other=0.0)

            # Apply RMSNorm
            x1_norm = x1.to(tl.float32) * rrms * w1.to(tl.float32)
            x2_norm = x2.to(tl.float32) * rrms * w2.to(tl.float32)

            # Load cos and sin
            cos_val = tl.load(cos_row + cols * stride_cos_dim, mask=mask, other=1.0)
            sin_val = tl.load(sin_row + cols * stride_cos_dim, mask=mask, other=0.0)

            # Apply RoPE rotation
            out1 = x1_norm * cos_val.to(tl.float32) - x2_norm * sin_val.to(tl.float32)
            out2 = x1_norm * sin_val.to(tl.float32) + x2_norm * cos_val.to(tl.float32)

            # Store results
            tl.store(out_row + idx1 * stride_out_hidden, out1.to(x1.dtype), mask=mask)
            tl.store(out_row + idx2 * stride_out_hidden, out2.to(x1.dtype), mask=mask)


def fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
    num_heads: int = None,
) -> torch.Tensor:
    """Apply fused RMSNorm + RoPE transformation.

    This function combines RMSNorm and Rotary Position Embedding into a single
    kernel launch, reducing memory bandwidth requirements.

    Args:
        x: Input tensor [batch, seq_len, hidden_dim]
        weight: RMSNorm weight [hidden_dim]
        cos: Cosine position embeddings [seq_len, head_dim]
        sin: Sine position embeddings [seq_len, head_dim]
        eps: Small constant for numerical stability
        num_heads: Number of attention heads (inferred if not provided)

    Returns:
        Output tensor [batch, seq_len, hidden_dim] with RMSNorm + RoPE applied
    """
    # Validate inputs
    batch_size, seq_len, hidden_dim, head_dim, num_heads = validate_rmsnorm_rope_inputs(
        x, weight, cos, sin, num_heads
    )

    # Handle 4D cos/sin format
    if cos.dim() == 4:
        cos = cos.squeeze(0).squeeze(1)  # [seq_len, head_dim]
        sin = sin.squeeze(0).squeeze(1)

    # Allocate output
    output = torch.empty_like(x)

    # Launch kernel
    grid = (batch_size * seq_len,)
    BLOCK_SIZE = min(128, head_dim // 2)

    fused_rmsnorm_rope_kernel[grid](
        x,
        output,
        weight,
        cos,
        sin,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        cos.stride(0),
        cos.stride(1),
        batch_size,
        seq_len,
        hidden_dim,
        head_dim,
        num_heads,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Reference implementation of RMSNorm for testing.

    Args:
        x: Input tensor [batch, seq_len, hidden_dim]
        weight: Weight tensor [hidden_dim]
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor
    """
    # Compute RMS
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    # Normalize and apply weight
    return (x.float() / rms * weight.float()).to(x.dtype)


def rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Reference implementation of RoPE for testing.

    Args:
        x: Input tensor [batch, seq_len, hidden_dim]
        cos: Cosine embeddings [seq_len, head_dim]
        sin: Sine embeddings [seq_len, head_dim]

    Returns:
        Tensor with RoPE applied
    """
    batch, seq_len, hidden_dim = x.shape
    head_dim = cos.shape[-1]
    num_heads = hidden_dim // head_dim

    # Reshape for head-wise processing
    x = x.view(batch, seq_len, num_heads, head_dim)

    # Split into two halves
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]

    # Expand cos/sin for broadcasting
    cos = cos[:seq_len, : head_dim // 2].unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim//2]
    sin = sin[:seq_len, : head_dim // 2].unsqueeze(0).unsqueeze(2)

    # Apply rotation
    out1 = x1.float() * cos.float() - x2.float() * sin.float()
    out2 = x1.float() * sin.float() + x2.float() * cos.float()

    # Concatenate and reshape back
    out = torch.cat([out1, out2], dim=-1).to(x.dtype)
    return out.view(batch, seq_len, hidden_dim)


def fused_rmsnorm_rope_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference implementation of fused RMSNorm + RoPE for testing.

    Args:
        x: Input tensor [batch, seq_len, hidden_dim]
        weight: RMSNorm weight [hidden_dim]
        cos: Cosine embeddings [seq_len, head_dim]
        sin: Sine embeddings [seq_len, head_dim]
        eps: Small constant for numerical stability

    Returns:
        Output tensor with RMSNorm + RoPE applied
    """
    # Handle 4D cos/sin format
    if cos.dim() == 4:
        cos = cos.squeeze(0).squeeze(1)
        sin = sin.squeeze(0).squeeze(1)

    # Apply RMSNorm first
    x_norm = rmsnorm_reference(x, weight, eps)
    # Then apply RoPE
    return rope_reference(x_norm, cos, sin)


class FusedRMSNormRoPE(torch.nn.Module):
    """PyTorch module wrapper for fused RMSNorm + RoPE.

    This module provides a convenient interface for using the fused kernel
    in PyTorch models.

    Args:
        hidden_dim: Hidden dimension size
        head_dim: Head dimension for RoPE
        eps: Small constant for numerical stability
    """

    def __init__(self, hidden_dim: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.eps = eps
        self.num_heads = hidden_dim // head_dim

        # RMSNorm weight parameter
        self.weight = torch.nn.Parameter(torch.ones(hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Apply fused RMSNorm + RoPE.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            cos: Cosine position embeddings
            sin: Sine position embeddings

        Returns:
            Output tensor with RMSNorm + RoPE applied
        """
        return fused_rmsnorm_rope(x, self.weight, cos, sin, self.eps, self.num_heads)
