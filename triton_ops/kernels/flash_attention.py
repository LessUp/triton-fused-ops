"""Triton FlashAttention 前向 kernel。

> ℹ️ **定位**：本实现是 [cuflash-attn](https://github.com/open-infra-ai/cuflash-attn)
> 的独立参考实现，用于验证 CUDA C++ 版本的正确性。完整 FlashAttention 前后向 +
> 优化叙事见 cuflash-attn（本仓库不承担完整 FA 交付物）。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton_ops.validation import validate_flash_attention_inputs


@triton.jit
def _flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    n_ctx,
    num_heads,
    stride_q_batch,
    stride_q_head,
    stride_q_seq,
    stride_q_dim,
    stride_k_batch,
    stride_k_head,
    stride_k_seq,
    stride_k_dim,
    stride_v_batch,
    stride_v_head,
    stride_v_seq,
    stride_v_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_seq,
    stride_out_dim,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    causal: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // num_heads
    head = batch_head % num_heads

    query_offsets = query_block * block_m + tl.arange(0, block_m)
    dim_offsets = tl.arange(0, head_dim)
    q_ptrs = (
        q_ptr
        + batch * stride_q_batch
        + head * stride_q_head
        + query_offsets[:, None] * stride_q_seq
        + dim_offsets[None, :] * stride_q_dim
    )
    q = tl.load(q_ptrs, mask=query_offsets[:, None] < n_ctx, other=0.0)

    accumulator = tl.zeros([block_m, head_dim], dtype=tl.float32)
    row_max = tl.full([block_m], float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros([block_m], dtype=tl.float32)
    scale = 1.0 / tl.sqrt(head_dim * 1.0)

    loop_end = tl.minimum((query_block + 1) * block_m, n_ctx) if causal else n_ctx
    for key_start in tl.range(0, loop_end, block_n):
        key_offsets = key_start + tl.arange(0, block_n)
        k_ptrs = (
            k_ptr
            + batch * stride_k_batch
            + head * stride_k_head
            + key_offsets[:, None] * stride_k_seq
            + dim_offsets[None, :] * stride_k_dim
        )
        v_ptrs = (
            v_ptr
            + batch * stride_v_batch
            + head * stride_v_head
            + key_offsets[:, None] * stride_v_seq
            + dim_offsets[None, :] * stride_v_dim
        )
        k = tl.load(k_ptrs, mask=key_offsets[:, None] < n_ctx, other=0.0)
        v = tl.load(v_ptrs, mask=key_offsets[:, None] < n_ctx, other=0.0)

        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(key_offsets[None, :] < n_ctx, scores, float("-inf"))
        if causal:
            scores = tl.where(
                query_offsets[:, None] >= key_offsets[None, :],
                scores,
                float("-inf"),
            )

        new_max = tl.maximum(row_max, tl.max(scores, axis=1))
        previous_scale = tl.exp(row_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        new_sum = previous_scale * row_sum + tl.sum(probabilities, axis=1)
        accumulator = previous_scale[:, None] * accumulator + tl.dot(probabilities.to(v.dtype), v)
        row_max = new_max
        row_sum = new_sum

    accumulator /= row_sum[:, None]
    out_ptrs = (
        out_ptr
        + batch * stride_out_batch
        + head * stride_out_head
        + query_offsets[:, None] * stride_out_seq
        + dim_offsets[None, :] * stride_out_dim
    )
    tl.store(out_ptrs, accumulator, mask=query_offsets[:, None] < n_ctx)


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """计算 `softmax(QK^T / sqrt(d))V`，不物化完整注意力矩阵。"""
    batch, heads, seq_len, head_dim = validate_flash_attention_inputs(q, k, v)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    output = torch.empty_like(q)

    block_m = 64
    block_n = 64
    grid = (triton.cdiv(seq_len, block_m), batch * heads)
    _flash_attention_kernel[grid](
        q,
        k,
        v,
        output,
        seq_len,
        heads,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        causal=causal,
    )
    return output
