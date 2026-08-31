"""用于 CPU 测试和 GPU 差分测试的参考实现。"""

from trifuse.reference.base import (
    Backend,
    BackendDispatcher,
    ensure_numpy,
    ensure_torch,
    reference_impl,
    to_output_dtype,
    validate_backend,
)
from trifuse.reference.flash_attention import flash_attention
from trifuse.reference.gated_mlp import gated_mlp
from trifuse.reference.rmsnorm_rope import fused_rmsnorm_rope, rmsnorm, rope

__all__ = [
    "rmsnorm",
    "rope",
    "fused_rmsnorm_rope",
    "gated_mlp",
    "flash_attention",
    "Backend",
    "validate_backend",
    "ensure_numpy",
    "ensure_torch",
    "to_output_dtype",
    "reference_impl",
    "BackendDispatcher",
]
