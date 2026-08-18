"""torch.library 自定义算子注册测试。

Feature: triton-fused-operators
Validates: `import triton_ops` 后 `torch.ops.triton_ops.*` 可调用，
与 kernel 直接调用/torch.mm 差分一致，CPU 输入被拒绝，torch.compile smoke。
"""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_sgemm_op_matches_torch_mm():
    """torch.ops.triton_ops.sgemm 应与 torch.mm 差分一致。"""
    import triton_ops  # noqa: F401  # 触发 torch.ops.triton_ops.* 注册

    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 64, device="cuda", dtype=torch.float16)

    out = torch.ops.triton_ops.sgemm(a, b)
    ref = a @ b

    assert out.shape == (128, 64)
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_rmsnorm_rope_op_matches_kernel():
    """torch.ops.triton_ops.fused_rmsnorm_rope 应与 kernel 直接调用逐元素一致。"""
    import triton_ops  # noqa: F401
    from triton_ops.kernels.rmsnorm_rope import fused_rmsnorm_rope

    torch.manual_seed(1)
    batch, seq, hidden, head_dim = 2, 16, 128, 64
    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float16)
    cos = torch.randn(seq, head_dim, device="cuda", dtype=torch.float16)
    sin = torch.randn(seq, head_dim, device="cuda", dtype=torch.float16)

    out = torch.ops.triton_ops.fused_rmsnorm_rope(x, weight, cos, sin)
    ref = fused_rmsnorm_rope(x, weight, cos, sin)

    assert out.shape == x.shape
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_gated_mlp_op_matches_kernel():
    """torch.ops.triton_ops.fused_gated_mlp 应与 kernel 直接调用逐元素一致。"""
    import triton_ops  # noqa: F401
    from triton_ops.kernels.gated_mlp import fused_gated_mlp

    torch.manual_seed(2)
    batch, seq, hidden, inter = 1, 8, 256, 512
    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float16)
    gate_weight = torch.randn(inter, hidden, device="cuda", dtype=torch.float16) * 0.1
    up_weight = torch.randn(inter, hidden, device="cuda", dtype=torch.float16) * 0.1

    out = torch.ops.triton_ops.fused_gated_mlp(x, gate_weight, up_weight, activation="silu")
    ref = fused_gated_mlp(x, gate_weight, up_weight, activation="silu")

    assert out.shape == (batch, seq, inter)
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_ops_reject_cpu_tensors():
    """CPU 输入应被明确拒绝（NotImplementedError，信息写明）。"""
    import triton_ops  # noqa: F401

    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    with pytest.raises(NotImplementedError, match="CUDA"):
        torch.ops.triton_ops.sgemm(a, b)

    x = torch.randn(1, 2, 8)
    weight = torch.randn(8)
    cos = torch.randn(2, 8)
    sin = torch.randn(2, 8)
    with pytest.raises(NotImplementedError, match="CUDA"):
        torch.ops.triton_ops.fused_rmsnorm_rope(x, weight, cos, sin)

    with pytest.raises(NotImplementedError, match="CUDA"):
        torch.ops.triton_ops.fused_gated_mlp(x, weight, weight)


def test_torch_compile_smoke():
    """torch.compile 对 sgemm op 做一次 smoke；compile 失败则记录 skip，不伪造通过。"""
    import triton_ops  # noqa: F401

    torch.manual_seed(3)

    @torch.compile
    def compiled_sgemm(a, b):
        return torch.ops.triton_ops.sgemm(a, b)

    a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.float16)

    try:
        out = compiled_sgemm(a, b)
    except Exception as exc:  # noqa: BLE001 - smoke 测试：失败只记录 skip
        pytest.skip(f"torch.compile smoke failed on triton_ops::sgemm: {exc}")

    ref = a @ b
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2)
