import pytest
import torch

from triton_ops import reference_flash_attention
from triton_ops.exceptions import ShapeMismatchError
from triton_ops.validation import validate_flash_attention_inputs


@pytest.mark.parametrize("causal", [False, True])
def test_reference_matches_pytorch_sdpa(causal: bool):
    torch.manual_seed(7)
    q = torch.randn(2, 3, 11, 16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = reference_flash_attention(q, k, v, causal=causal)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

    torch.testing.assert_close(actual, expected)


def test_validation_rejects_shape_mismatch_before_device_check():
    q = torch.empty(1, 2, 8, 16)
    k = torch.empty(1, 2, 7, 16)
    v = torch.empty_like(q)

    with pytest.raises(ShapeMismatchError, match="shapes must match"):
        validate_flash_attention_inputs(q, k, v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seq_len", [11, 65])
def test_kernel_matches_reference_for_tail_blocks(causal: bool, seq_len: int):
    from triton_ops.kernels.flash_attention import flash_attention

    torch.manual_seed(11)
    q = torch.randn(2, 3, seq_len, 16, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = flash_attention(q, k, v, causal=causal)
    expected = reference_flash_attention(q, k, v, causal=causal)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def _compiled_kernel_count(kernel):
    total = 0
    for caches in kernel.device_caches.values():
        total += len(caches[0])
    return total


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kernel_not_recompiled_per_seq_len():
    """不同 seq_len 首次调用不应触发 kernel 重新编译（维度/stride 不能全部 constexpr）。"""
    from triton_ops.kernels.flash_attention import _flash_attention_kernel, flash_attention

    torch.manual_seed(3)
    base = _compiled_kernel_count(_flash_attention_kernel)
    for seq_len in (256, 128, 64):
        q = torch.randn(1, 2, seq_len, 64, device="cuda", dtype=torch.float16)
        k, v = torch.randn_like(q), torch.randn_like(q)
        flash_attention(q, k, v, causal=False)
    new_kernels = _compiled_kernel_count(_flash_attention_kernel) - base
    assert new_kernels <= 1, f"每个 seq_len 触发重新编译：新增 {new_kernels} 个 kernel"
