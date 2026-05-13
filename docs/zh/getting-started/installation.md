---
title: 安装指南
description: "Triton Fused Ops 的环境要求与安装流程"
---

# 安装指南

本页用于准备可工作的运行环境，并完成最基本的验证。

## 基础要求

| 项目 | 基线 | 说明 |
|:--|:--|:--|
| Python | `>=3.9` | 来自包元数据约束 |
| PyTorch | `>=2.0.0` | 真正执行 Triton kernel 需要 CUDA 版 PyTorch |
| Triton | `>=2.1.0` | OpenAI Triton |
| GPU | NVIDIA CUDA GPU | 运行 kernel 和 GPU benchmark 需要 |

## 从源码安装

```bash
git clone https://github.com/LessUp/triton-fused-ops.git
cd triton-fused-ops
pip install -e ".[dev]"
```

如果你只需要包本体：

```bash
pip install -e .
```

如果使用 `uv`：

```bash
uv pip install -e ".[dev]"
```

## CPU-safe 基线检查

下面这些命令不要求真正运行 Triton kernel，适合 CI 或仅 CPU 的验证路径：

```bash
python -c "import triton_ops; print(triton_ops.__version__)"
ruff format --check .
ruff check .
mypy triton_ops/
pytest tests/ -v -k "not cuda and not gpu" --ignore=tests/benchmarks/
python3 -m build
```

## GPU 冒烟测试

```python
import torch
from triton_ops import fused_rmsnorm_rope

assert torch.cuda.is_available()

batch, seq_len, hidden_dim, head_dim = 2, 128, 4096, 64
x = torch.randn(batch, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
weight = torch.ones(hidden_dim, device="cuda", dtype=torch.float16)
cos = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)
sin = torch.randn(seq_len, head_dim, device="cuda", dtype=torch.float16)

y = fused_rmsnorm_rope(x, weight, cos, sin)
print(y.shape, y.dtype)
```

说明：

- 当前实现接受 `[seq_len, head_dim]` 形状的 `cos` / `sin`。
- 验证层也接受 `[1, seq_len, 1, head_dim]` 形状的 4D RoPE cache。
- 运行时要求输入在 CUDA 上、dtype 合法且内存连续。

## 环境检查

```python
import torch

print("CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA 版本:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name())
```

## 常见问题

### `CUDA is not available`

常见原因：

- 当前 PyTorch 不是 CUDA 版本。
- 当前 Python 环境看不到对应的 NVIDIA 驱动或运行时。

常见修复：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 调用 kernel 时出现 `DeviceError`

导出的 kernel 会先检查输入是否位于 CUDA。请确保所有输入在同一个 CUDA 设备上，并且保持 contiguous。

### 出现 `UnsupportedDtypeError` 或 shape 校验失败

请直接对照 API 页中的输入契约：

- `fused_rmsnorm_rope`：3D `x`、1D `weight`、2D 或 4D RoPE cache。
- `fused_gated_mlp`：3D `x`、2D 权重、激活函数只能是 `"silu"` 或 `"gelu"`。
- `fp8_gemm`：2D 矩阵；若输入已经是预量化 FP8，则必须提供对应 scale。

## 下一步

- [快速开始](/zh/getting-started/quickstart)
- [示例教程](/zh/getting-started/examples)
- [核心算子 API](/zh/api/kernels)
