"""torch.library 自定义算子注册测试。

Feature: triton-fused-operators
Validates: `import trifuse` 后 `torch.ops.trifuse.*` 可调用，
与 kernel 直接调用/torch.mm 差分一致，CPU 输入被拒绝，torch.compile smoke。
"""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_sgemm_op_matches_torch_mm():
    """torch.ops.trifuse.sgemm 应与 torch.mm 差分一致。"""
    import trifuse  # noqa: F401  # 触发 torch.ops.trifuse.* 注册

    torch.manual_seed(0)
    a = torch.randn(128, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 64, device="cuda", dtype=torch.float16)

    out = torch.ops.trifuse.sgemm(a, b)
    ref = a @ b

    assert out.shape == (128, 64)
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_rmsnorm_rope_op_matches_kernel():
    """torch.ops.trifuse.fused_rmsnorm_rope 应与 kernel 直接调用逐元素一致。"""
    import trifuse  # noqa: F401
    from trifuse.kernels.rmsnorm_rope import fused_rmsnorm_rope

    torch.manual_seed(1)
    batch, seq, hidden, head_dim = 2, 16, 128, 64
    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float16)
    cos = torch.randn(seq, head_dim, device="cuda", dtype=torch.float16)
    sin = torch.randn(seq, head_dim, device="cuda", dtype=torch.float16)

    out = torch.ops.trifuse.fused_rmsnorm_rope(x, weight, cos, sin)
    ref = fused_rmsnorm_rope(x, weight, cos, sin)

    assert out.shape == x.shape
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_gated_mlp_op_matches_kernel():
    """torch.ops.trifuse.fused_gated_mlp 应与 kernel 直接调用逐元素一致。"""
    import trifuse  # noqa: F401
    from trifuse.kernels.gated_mlp import fused_gated_mlp

    torch.manual_seed(2)
    batch, seq, hidden, inter = 1, 8, 256, 512
    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float16)
    gate_weight = torch.randn(inter, hidden, device="cuda", dtype=torch.float16) * 0.1
    up_weight = torch.randn(inter, hidden, device="cuda", dtype=torch.float16) * 0.1

    out = torch.ops.trifuse.fused_gated_mlp(x, gate_weight, up_weight, activation="silu")
    ref = fused_gated_mlp(x, gate_weight, up_weight, activation="silu")

    assert out.shape == (batch, seq, inter)
    assert torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


def test_ops_reject_cpu_tensors():
    """CPU 输入应被明确拒绝（NotImplementedError，信息写明）。"""
    import trifuse  # noqa: F401

    a = torch.randn(4, 4)
    b = torch.randn(4, 4)
    with pytest.raises(NotImplementedError, match="CUDA"):
        torch.ops.trifuse.sgemm(a, b)

    x = torch.randn(1, 2, 8)
    weight = torch.randn(8)
    cos = torch.randn(2, 8)
    sin = torch.randn(2, 8)
    with pytest.raises(NotImplementedError, match="CUDA"):
        torch.ops.trifuse.fused_rmsnorm_rope(x, weight, cos, sin)

    with pytest.raises(NotImplementedError, match="CUDA"):
        torch.ops.trifuse.fused_gated_mlp(x, weight, weight)


def test_torch_compile_smoke():
    """torch.compile 对三个公开 op 做 smoke：必须编译成功且结果正确。"""
    import trifuse  # noqa: F401
    from trifuse.reference import fused_rmsnorm_rope as ref_rmsnorm_rope
    from trifuse.reference import gated_mlp as ref_gated_mlp

    torch.manual_seed(3)

    @torch.compile
    def compiled_sgemm(a, b):
        return torch.ops.trifuse.sgemm(a, b)

    @torch.compile
    def compiled_rmsnorm_rope(x, weight, cos, sin):
        return torch.ops.trifuse.fused_rmsnorm_rope(x, weight, cos, sin)

    @torch.compile
    def compiled_gated_mlp(x, gate_w, up_w):
        return torch.ops.trifuse.fused_gated_mlp(x, gate_w, up_w, activation="silu")

    a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    out = compiled_sgemm(a, b)
    assert torch.allclose(out.float(), (a @ b).float(), rtol=1e-2, atol=1e-2)

    batch, seq, hidden, head_dim = 1, 8, 128, 64
    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden, device="cuda", dtype=torch.float16)
    cos = torch.randn(seq, head_dim, device="cuda", dtype=torch.float16)
    sin = torch.randn(seq, head_dim, device="cuda", dtype=torch.float16)
    out_rn = compiled_rmsnorm_rope(x, weight, cos, sin)
    ref_rn = ref_rmsnorm_rope(x, weight, cos, sin, backend="cuda")
    assert torch.allclose(out_rn.float(), ref_rn.float(), rtol=2e-2, atol=1e-2)

    inter = 256
    gw = torch.randn(inter, hidden, device="cuda", dtype=torch.float16) * 0.1
    uw = torch.randn(inter, hidden, device="cuda", dtype=torch.float16) * 0.1
    out_gm = compiled_gated_mlp(x, gw, uw)
    ref_gm = ref_gated_mlp(x, gw, uw, activation="silu", backend="cuda")
    assert out_gm.shape == (batch, seq, inter)
    assert torch.allclose(out_gm.float(), ref_gm.float(), rtol=1e-2, atol=1e-2)
