"""torch.library 自定义算子注册。

把 `triton_ops.kernels.*` 的公开函数注册进 `torch.ops.triton_ops.*` 命名空间，
与 vLLM/SGLang 等推理框架用 `torch.library` 注册自定义 op 的方式一致：

- `triton_ops::sgemm(a, b) -> Tensor`
- `triton_ops::fused_rmsnorm_rope(x, weight, cos, sin, eps=1e-6) -> Tensor`
- `triton_ops::fused_gated_mlp(x, gate_weight, up_weight, activation="silu") -> Tensor`

注册策略：统一用 `torch.library.custom_op + register_fake`（torch>=2.4）：
- `custom_op` 提供 eager 执行；
- `register_fake` 提供 shape 推断（fake 实现），使 op 对 torch.compile / torch.export
  视为 opaque 自定义算子，可编译、可导出。

为什么不选 `torch.library.triton_op`：triton_op 对 Dynamo/Inductor 是透明的，会追踪进
实现体并尝试访问 fake tensor 的 data pointer，实测在 torch 2.13 下 torch.compile 报
"Cannot access data pointer of Tensor"。要让 triton_op 可用，需把 kernel 用
`torch._library.triton.wrap_triton` 注册（本仓库不承担该集成），故退回 opaque 方案。

所有 op 只接受 CUDA 张量；CPU 输入直接抛 `NotImplementedError` 并写明原因。
op 内部只调用 `triton_ops.kernels.*` 的公开函数，不复制 kernel 逻辑。
"""

import torch

from triton_ops.kernels.gated_mlp import fused_gated_mlp
from triton_ops.kernels.rmsnorm_rope import fused_rmsnorm_rope
from triton_ops.kernels.sgemm import sgemm


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
    """注册一个自定义算子到 `torch.ops`（torch>=2.4）。

    使用 `torch.library.custom_op + register_fake`：eager 执行走 `impl`，
    `register_fake` 提供 shape 推断，使 torch.compile / torch.export 可将其
    视为 opaque 自定义算子处理。返回 op 定义。
    """
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
