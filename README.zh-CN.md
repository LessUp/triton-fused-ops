# Triton Fused Ops

<div align="center">

[![CI](https://github.com/LessUp/triton-fused-ops/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/triton-fused-ops/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/triton-fused-ops/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/triton-fused-ops/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Triton 2.1+](https://img.shields.io/badge/Triton-2.1+-76B900?logo=nvidia&logoColor=white)

面向 Transformer 推理场景的高性能 Triton 算子集合。

[📖 文档](https://lessup.github.io/triton-fused-ops/) | [🇺🇸 English](README.md) | [💡 示例](examples/) | [🤝 贡献指南](CONTRIBUTING.md)

</div>

---

## 仓库提供的能力

`triton-fused-ops` 当前聚焦三类核心算子：

- `fused_rmsnorm_rope`：RMSNorm + RoPE 融合
- `fused_gated_mlp`：Gated MLP 融合（SiLU/GELU）
- `fp8_gemm` 及量化工具：FP8 矩阵乘法链路

目标是降低冗余显存访问，并保持可测试、可集成的接口。

## 运行边界

- **GPU 必需**：Triton kernel 执行与性能基准测试必须使用 CUDA GPU。
- **CPU 环境可验证**：可运行导入/类型检查/lint/CPU-safe 测试（CI 采用此路径）。
- 性能结果与 GPU 架构、模型形状、batch 大小强相关。

## 安装

```bash
git clone https://github.com/LessUp/triton-fused-ops.git
cd triton-fused-ops
pip install -e ".[dev]"
```

## 快速验证

CPU 环境下导入验证：

```bash
python -c "import triton_ops; print(triton_ops.__version__)"
```

仓库基线检查：

```bash
ruff format --check .
ruff check .
mypy triton_ops/
pytest tests/ -v -k "not cuda and not gpu" --ignore=tests/benchmarks/
python3 -m build
```

## 最小用例（GPU）

```python
import torch
from triton_ops import fused_rmsnorm_rope

x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
weight = torch.ones(4096, device="cuda", dtype=torch.float16)
cos = torch.randn(128, 128, device="cuda", dtype=torch.float16)
sin = torch.randn(128, 128, device="cuda", dtype=torch.float16)

y = fused_rmsnorm_rope(x, weight, cos, sin)
print(y.shape)
```

## 性能说明

仓库中的代表性数据来自 NVIDIA A100（CUDA 12.1），用于说明趋势，不代表所有硬件/场景下的统一结论。

| 算子 | 相对参考路径的典型加速区间 |
|:--|:--:|
| `fused_rmsnorm_rope` | 最高约 ~3x |
| `fused_gated_mlp` | 约 ~1.3x–1.8x |
| `fp8_gemm` | 约 ~1.2x–1.5x |

详见 `tests/benchmarks/` 与文档说明。

## 开发流程

本仓库对非简单改动采用 **OpenSpec 驱动**：

1. 创建/选择 OpenSpec change
2. 完成 proposal/design/specs/tasks
3. 按任务顺序实施
4. 审查并通过验证后合并

参阅：

- [`AGENTS.md`](AGENTS.md)
- [`CLAUDE.md`](CLAUDE.md)
- [`openspec/README.md`](openspec/README.md)

## 许可证

MIT。
