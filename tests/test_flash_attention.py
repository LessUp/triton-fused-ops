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
