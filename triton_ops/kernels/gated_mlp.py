"""Fused Gated MLP Triton kernel implementation.

This module implements a fused kernel for Gated MLP (used in LLaMA, Mistral, etc.)
that combines gate projection, up projection, and activation into a single kernel.

Mathematical formula:
output = gate_proj(x) * activation(up_proj(x))

Where activation is either SiLU (x * sigmoid(x)) or GELU.
"""

import torch
import triton
import triton.language as tl

from triton_ops.validation import validate_gated_mlp_inputs


@triton.jit
def silu(x):
    """SiLU (Swish) activation: x * sigmoid(x)"""
    return x * tl.sigmoid(x)


@triton.jit
def gelu(x):
    """GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))"""
    return x * 0.5 * (1.0 + tl.libdevice.erf(x * 0.7071067811865476))


@triton.jit
def fused_gated_mlp_kernel(
    x_ptr,
    gate_weight_ptr,
    up_weight_ptr,
    output_ptr,
    # Strides for x [batch, seq_len, hidden_dim]
    stride_x_batch,
    stride_x_seq,
    stride_x_hidden,
    # Strides for weights [intermediate_dim, hidden_dim]
    stride_gw_inter,
    stride_gw_hidden,
    stride_uw_inter,
    stride_uw_hidden,
    # Strides for output [batch, seq_len, intermediate_dim]
    stride_out_batch,
    stride_out_seq,
    stride_out_inter,
    # Dimensions
    batch_size,
    seq_len,
    hidden_dim,
    intermediate_dim,
    # Activation type: 0=SiLU, 1=GELU
    activation_type: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused Gated MLP kernel.
    
    Computes: output = gate_proj(x) * activation(up_proj(x))
    
    Each program computes a BLOCK_M x BLOCK_N tile of the output.
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Compute which tile this program handles
    num_tiles_m = tl.cdiv(batch_size * seq_len, BLOCK_M)
    num_tiles_n = tl.cdiv(intermediate_dim, BLOCK_N)
    
    tile_m = pid // num_tiles_n
    tile_n = pid % num_tiles_n
    
    # Compute row and column ranges for this tile
    row_start = tile_m * BLOCK_M
    col_start = tile_n * BLOCK_N
    
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)
    
    # Compute batch and seq indices from row index
    batch_indices = rows // seq_len
    seq_indices = rows % seq_len
    
    # Initialize accumulators for gate and up projections
    gate_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    up_acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Compute matrix multiplication in blocks
    for k_start in range(0, hidden_dim, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_range < hidden_dim
        
        # Load x block [BLOCK_M, BLOCK_K]
        x_ptrs = (x_ptr + 
                  batch_indices[:, None] * stride_x_batch + 
                  seq_indices[:, None] * stride_x_seq + 
                  k_range[None, :] * stride_x_hidden)
        row_mask = rows[:, None] < (batch_size * seq_len)
        x_mask = row_mask & k_mask[None, :]
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load gate weight block [BLOCK_K, BLOCK_N]
        gw_ptrs = (gate_weight_ptr + 
                   cols[None, :] * stride_gw_inter + 
                   k_range[:, None] * stride_gw_hidden)
        col_mask = cols[None, :] < intermediate_dim
        gw_mask = k_mask[:, None] & col_mask
        gw_block = tl.load(gw_ptrs, mask=gw_mask, other=0.0)
        
        # Load up weight block [BLOCK_K, BLOCK_N]
        uw_ptrs = (up_weight_ptr + 
                   cols[None, :] * stride_uw_inter + 
                   k_range[:, None] * stride_uw_hidden)
        uw_block = tl.load(uw_ptrs, mask=gw_mask, other=0.0)
        
        # Accumulate matrix products
        gate_acc += tl.dot(x_block.to(tl.float32), gw_block.to(tl.float32))
        up_acc += tl.dot(x_block.to(tl.float32), uw_block.to(tl.float32))
    
    # Apply activation to up projection
    if activation_type == 0:
        up_activated = silu(up_acc)
    else:
        up_activated = gelu(up_acc)
    
    # Compute gated output
    output = gate_acc * up_activated
    
    # Store output
    out_ptrs = (output_ptr + 
                batch_indices[:, None] * stride_out_batch + 
                seq_indices[:, None] * stride_out_seq + 
                cols[None, :] * stride_out_inter)
    out_mask = (rows[:, None] < (batch_size * seq_len)) & (cols[None, :] < intermediate_dim)
    
    # Convert to output dtype
    tl.store(out_ptrs, output.to(tl.float16), mask=out_mask)


def fused_gated_mlp(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """Apply fused Gated MLP transformation.
    
    Computes: output = gate_proj(x) * activation(up_proj(x))
    
    This fused implementation reduces memory bandwidth by computing both
    projections and the activation in a single kernel.
    
    Args:
        x: Input tensor [batch, seq_len, hidden_dim]
        gate_weight: Gate projection weight [intermediate_dim, hidden_dim]
        up_weight: Up projection weight [intermediate_dim, hidden_dim]
        activation: Activation function ("silu" or "gelu")
        
    Returns:
        Output tensor [batch, seq_len, intermediate_dim]
    """
    # Validate inputs
    batch_size, seq_len, hidden_dim, intermediate_dim = validate_gated_mlp_inputs(
        x, gate_weight, up_weight, activation
    )
    
    # Allocate output
    output = torch.empty(
        batch_size, seq_len, intermediate_dim,
        dtype=x.dtype, device=x.device
    )
    
    # Determine activation type
    activation_type = 0 if activation == "silu" else 1
    
    # Block sizes
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 32
    
    # Grid size
    num_tiles_m = triton.cdiv(batch_size * seq_len, BLOCK_M)
    num_tiles_n = triton.cdiv(intermediate_dim, BLOCK_N)
    grid = (num_tiles_m * num_tiles_n,)
    
    # Launch kernel
    fused_gated_mlp_kernel[grid](
        x, gate_weight, up_weight, output,
        x.stride(0), x.stride(1), x.stride(2),
        gate_weight.stride(0), gate_weight.stride(1),
        up_weight.stride(0), up_weight.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        batch_size, seq_len, hidden_dim, intermediate_dim,
        activation_type=activation_type,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return output


def gated_mlp_reference(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """Reference implementation of Gated MLP for testing.
    
    Args:
        x: Input tensor [batch, seq_len, hidden_dim]
        gate_weight: Gate projection weight [intermediate_dim, hidden_dim]
        up_weight: Up projection weight [intermediate_dim, hidden_dim]
        activation: Activation function ("silu" or "gelu")
        
    Returns:
        Output tensor [batch, seq_len, intermediate_dim]
    """
    # Compute projections
    gate = torch.nn.functional.linear(x.float(), gate_weight.float())
    up = torch.nn.functional.linear(x.float(), up_weight.float())
    
    # Apply activation
    if activation == "silu":
        up_activated = torch.nn.functional.silu(up)
    else:
        up_activated = torch.nn.functional.gelu(up)
    
    # Gated output
    output = gate * up_activated
    
    return output.to(x.dtype)


class FusedGatedMLP(torch.nn.Module):
    """PyTorch module wrapper for fused Gated MLP.
    
    This module provides a convenient interface for using the fused kernel
    in PyTorch models.
    
    Args:
        hidden_dim: Input hidden dimension
        intermediate_dim: Intermediate (FFN) dimension
        activation: Activation function ("silu" or "gelu")
    """
    
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        activation: str = "silu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.activation = activation
        
        # Weight parameters
        self.gate_weight = torch.nn.Parameter(
            torch.randn(intermediate_dim, hidden_dim) * 0.02
        )
        self.up_weight = torch.nn.Parameter(
            torch.randn(intermediate_dim, hidden_dim) * 0.02
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fused Gated MLP.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            
        Returns:
            Output tensor [batch, seq_len, intermediate_dim]
        """
        return fused_gated_mlp(x, self.gate_weight, self.up_weight, self.activation)
