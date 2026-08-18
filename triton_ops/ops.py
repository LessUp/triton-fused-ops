"""torch.library 自定义算子注册。

把 `triton_ops.kernels.*` 的公开函数注册进 `torch.ops.triton_ops.*` 命名空间，
与 vLLM/SGLang 等推理框架用 `torch.library` 注册自定义 op 的方式一致：

- `triton_ops::sgemm(a, b) -> Tensor`
- `triton_ops::fused_rmsnorm_rope(x, weight, cos, sin, eps=1e-6) -> Tensor`
- `triton_ops::fused_gated_mlp(x, gate_weight, up_weight, activation="silu") -> Tensor`

注册策略：
- 优先用 `torch.library.triton_op`（torch 2.13+）：实现由 Triton kernel 构成，
  torch.compile/export 可见并可优化；
- 不可用时 fallback 到 `torch.library.custom_op + register_fake`：把 op 当 opaque，
  提供 eager 执行与 shape 推断（fake 实现）。

所有 op 只接受 CUDA 张量；CPU 输入直接抛 `NotImplementedError` 并写明原因。
op 内部只调用 `triton_ops.kernels.*` 的公开函数，不复制 kernel 逻辑。
"""

import torch

from triton_ops.kernels.gated_mlp import fused_gated_mlp
from triton_ops.kernels.rmsnorm_rope import fused_rmsnorm_rope
from triton_ops.kernels.sgemm import sgemm

_HAS_TRITON_OP = hasattr(torch.library, "triton_op")


def _check_cuda(*tensors: torch.Tensor) -> None:
    """只允许 CUDA 张量；CPU 直接抛 NotImplementedError（信息写明）。"""
    for tensor in tensors:
        if not tensor.is_cuda:
            raise NotImplementedError(
                f"triton_ops custom op only supports CUDA tensors, got {tensor.device}"
            )


# ---------------------------------------------------------------------------
# eager 实现：只调用 triton_ops.kernels.* 的公开函数
# ---------------------------------------------------------------------------


def _sgemm_impl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _check_cuda(a, b)
    return sgemm(a, b)


def _fused_rmsnorm_rope_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    _check_cuda(x, weight, cos, sin)
    return fused_rmsnorm_rope(x, weight, cos, sin, eps=eps)


def _fused_gated_mlp_impl(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    _check_cuda(x, gate_weight, up_weight)
    return fused_gated_mlp(x, gate_weight, up_weight, activation=activation)


# ---------------------------------------------------------------------------
# fake 实现（fallback 路径的 shape 推断；shape 与 eager 一致）
# ---------------------------------------------------------------------------


def _fake_sgemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.new_empty((a.shape[0], b.shape[1]))


def _fake_fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return x.new_empty(x.shape)


def _fake_fused_gated_mlp(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    return x.new_empty(x.shape[:-1] + (gate_weight.shape[0],))


# ---------------------------------------------------------------------------
# 注册 helper
# ---------------------------------------------------------------------------


def _register(
    name: str,
    impl,
    fake_impl,
    mutates_args=(),
):
    """注册一个自定义算子到 `torch.ops`。

    优先 `torch.library.triton_op`；不可用则 fallback 到
    `torch.library.custom_op + register_fake`。返回 op 定义。
    """
    if _HAS_TRITON_OP:
        return torch.library.triton_op(name, mutates_args=mutates_args)(impl)
    op = torch.library.custom_op(name, mutates_args=mutates_args)(impl)
    torch.library.register_fake(name, fake_impl)
    return op


_register("triton_ops::sgemm", _sgemm_impl, _fake_sgemm)
_register(
    "triton_ops::fused_rmsnorm_rope",
    _fused_rmsnorm_rope_impl,
    _fake_fused_rmsnorm_rope,
)
_register(
    "triton_ops::fused_gated_mlp",
    _fused_gated_mlp_impl,
    _fake_fused_gated_mlp,
)
