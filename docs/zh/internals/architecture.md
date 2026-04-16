---
layout: default
title: "架构设计 — Triton Fused Ops"
description: "Triton Fused Ops 整体架构概览"
---

# 架构设计概览

理解 Triton Fused Ops 的整体架构。

---

## 高层设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        API 层                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    核心算子   │  │    自动调优   │  │    基准测试   │          │
│  │   (api.py)   │  │   (tuner)    │  │   (suite)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Triton Kernel 层                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐     │
│  │ rmsnorm_rope   │  │  gated_mlp     │  │   fp8_gemm     │     │
│  │   (triton)     │  │   (triton)     │  │   (triton)     │     │
│  └────────────────┘  └────────────────┘  └────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GPU 硬件                                   │
│              CUDA / PTX / SASS 指令集                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 模块组织

```
triton_ops/
├── __init__.py          # 公共 API 导出
├── api.py               # 简洁函数式 API
├── models.py            # 数据模型和类型
├── exceptions.py        # 自定义异常
├── utils.py             # 工具函数
├── kernels/             # Triton 算子实现
│   ├── rmsnorm_rope.py
│   ├── gated_mlp.py
│   ├── fp8_gemm.py
│   └── fp8_quantize.py
├── autotuner/           # 自动调优框架
│   ├── tuner.py
│   ├── configs.py
│   └── cache.py
└── benchmark/           # 基准测试套件
    ├── suite.py
    ├── correctness.py
    └── report.py
```

---

## 设计原则

### 1. 关注点分离

| 层级 | 职责 |
|:------|:---------------|
| **API 层** | 用户界面、输入验证 |
| **Kernel 层** | 底层 Triton 实现 |
| **调优层** | 配置优化 |
| **硬件层** | 实际计算 |

### 2. 延迟加载

算子和配置在首次使用时才加载：

```python
# 算子在首次调用前不会编译
from triton_ops import fused_rmsnorm_rope

# 首次调用触发 Triton JIT 编译
output = fused_rmsnorm_rope(x, weight, cos, sin)

# 后续调用使用缓存的二进制文件
```

### 3. 类型安全

全面的类型提示：

```python
def fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    ...
```

---

## 核心算子注册

```python
# triton_ops/__init__.py

# 核心算子
from .kernels.rmsnorm_rope import (
    fused_rmsnorm_rope,
    FusedRMSNormRoPE,
)
from .kernels.gated_mlp import (
    fused_gated_mlp,
    FusedGatedMLP,
)
from .kernels.fp8_gemm import (
    fp8_gemm,
    FP8Linear,
)
from .kernels.fp8_quantize import (
    quantize_fp8,
    dequantize_fp8,
    quantize_fp8_with_overflow_handling,
)

# 自动调优
from .autotuner.tuner import (
    TritonAutoTuner,
    ConfigCache,
)
from .autotuner.configs import (
    RMSNORM_ROPE_CONFIGS,
    GATED_MLP_CONFIGS,
    FP8_GEMM_CONFIGS,
)

__all__ = [
    # 核心算子
    "fused_rmsnorm_rope",
    "FusedRMSNormRoPE",
    "fused_gated_mlp",
    "FusedGatedMLP",
    "fp8_gemm",
    "FP8Linear",
    # 量化
    "quantize_fp8",
    "dequantize_fp8",
    "quantize_fp8_with_overflow_handling",
    # 自动调优
    "TritonAutoTuner",
    "ConfigCache",
    "RMSNORM_ROPE_CONFIGS",
    "GATED_MLP_CONFIGS",
    "FP8_GEMM_CONFIGS",
]
```

---

## 错误处理

### 异常层次结构

```
TritonOpsError (基类)
├── DeviceError
│   └── CUDA 不可用
├── ShapeMismatchError
│   └── 张量形状不兼容
├── DtypeError
│   └── 不支持的数据类型
├── TuningError
│   └── 自动调优失败
└── NumericalError
    └── 量化溢出
```

### 使用示例

```python
from triton_ops import fused_rmsnorm_rope, DeviceError, ShapeMismatchError

try:
    output = fused_rmsnorm_rope(x, weight, cos, sin)
except DeviceError as e:
    print(f"CUDA 错误: {e}")
except ShapeMismatchError as e:
    print(f"形状错误: {e}")
```

---

## 扩展点

### 添加新算子

1. 在 `kernels/` 中实现 Triton 算子
2. 在 `kernels/<name>.py` 中添加函数式 API
3. 添加模块封装（可选）
4. 在 `__init__.py` 中导出
5. 在 `autotuner/configs.py` 中添加配置空间
6. 在 `tests/` 中添加测试

### 添加自动调优支持

```python
# 在 autotuner/configs.py 中

MY_KERNEL_CONFIGS = {
    "BLOCK_M": [64, 128, 256],
    "BLOCK_N": [64, 128, 256],
    "num_warps": [4, 8],
}

# 在 __init__.py 中
from .autotuner.configs import MY_KERNEL_CONFIGS
__all__.append("MY_KERNEL_CONFIGS")
```

---

<div align="center">

**[⬆ 返回顶部](#架构设计概览)** | **[← 返回内部文档](../)**

</div>
