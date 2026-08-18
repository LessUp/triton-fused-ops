"""面向 Transformer 推理学习的精简 Triton 算子库。"""

from triton_ops.autotuner import (
    GATED_MLP_CONFIGS,
    RMSNORM_ROPE_CONFIGS,
    ConfigCache,
    TritonAutoTuner,
)
from triton_ops.benchmark import (
    BenchmarkSuite,
    CorrectnessVerifier,
    KernelBenchmark,
    PerformanceReport,
)
from triton_ops.exceptions import (
    DeviceError,
    ShapeMismatchError,
    TritonKernelError,
    TuningFailedError,
    UnsupportedDtypeError,
)
from triton_ops.models import KernelMetrics, TensorSpec, TuningResult
from triton_ops.performance import PerformanceProfile, compute_metrics
from triton_ops.reference import (
    flash_attention as reference_flash_attention,
)
from triton_ops.reference import (
    fused_rmsnorm_rope as reference_fused_rmsnorm_rope,
)
from triton_ops.reference import gated_mlp as reference_gated_mlp
from triton_ops.reference import rmsnorm as reference_rmsnorm
from triton_ops.reference import rope as reference_rope

__version__ = "2.0.0"

__all__ = [
    "PerformanceProfile",
    "compute_metrics",
    "TensorSpec",
    "KernelMetrics",
    "TuningResult",
    "TritonKernelError",
    "ShapeMismatchError",
    "UnsupportedDtypeError",
    "TuningFailedError",
    "DeviceError",
    "TritonAutoTuner",
    "ConfigCache",
    "RMSNORM_ROPE_CONFIGS",
    "GATED_MLP_CONFIGS",
    "BenchmarkSuite",
    "CorrectnessVerifier",
    "PerformanceReport",
    "KernelBenchmark",
    "reference_rmsnorm",
    "reference_rope",
    "reference_fused_rmsnorm_rope",
    "reference_gated_mlp",
    "reference_flash_attention",
]

try:
    from triton_ops.kernels import (
        FusedGatedMLP,
        FusedRMSNormRoPE,
        flash_attention,
        fused_gated_mlp,
        fused_rmsnorm_rope,
        sgemm,
    )
except ModuleNotFoundError as error:
    if error.name != "triton":
        raise
else:
    from triton_ops import ops as _ops  # noqa: F401  # 注册 torch.ops.triton_ops.*

    __all__ += [
        "fused_rmsnorm_rope",
        "FusedRMSNormRoPE",
        "fused_gated_mlp",
        "FusedGatedMLP",
        "flash_attention",
        "sgemm",
    ]
