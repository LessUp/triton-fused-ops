"""FlashAttention 的 PyTorch 参考实现。"""

from __future__ import annotations

import math

import torch


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """用显式 matmul、mask 和 softmax 计算独立参考结果。"""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(f"q, k and v must be 4D; got {q.dim()}D, {k.dim()}D and {v.dim()}D")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k and v shapes must match; got {q.shape}, {k.shape} and {v.shape}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError(f"q, k and v dtypes must match; got {q.dtype}, {k.dtype} and {v.dtype}")
    if q.device != k.device or q.device != v.device:
        raise ValueError(
            f"q, k and v devices must match; got {q.device}, {k.device} and {v.device}"
        )

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        seq_len = q.shape[-2]
        future = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device),
            diagonal=1,
        )
        scores = scores.masked_fill(future, float("-inf"))

    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v.float()).to(q.dtype)
