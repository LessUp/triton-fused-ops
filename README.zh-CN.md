# Triton Fused Ops

<div align="center">

[![CI](https://github.com/LessUp/triton-fused-ops/actions/workflows/ci.yml/badge.svg)](https://github.com/LessUp/triton-fused-ops/actions/workflows/ci.yml)
[![Pages](https://github.com/LessUp/triton-fused-ops/actions/workflows/pages.yml/badge.svg)](https://lessup.github.io/triton-fused-ops/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Triton 2.1+](https://img.shields.io/badge/Triton-2.1+-76B900?logo=nvidia&logoColor=white)
![CUDA 12.1+](https://img.shields.io/badge/CUDA-12.1+-76B900?logo=nvidia&logoColor=white)
![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-261230?logo=ruff&logoColor=white)
![Types: mypy](https://img.shields.io/badge/Types-mypy-2A6DB5?logo=python&logoColor=white)

**面向 LLM 推理的融合 GPU 算子库。Memory-bound &rarr; Compute-bound。**

[📖 文档](https://lessup.github.io/triton-fused-ops/) | [🇺🇸 English](README.md) | [💡 示例](examples/) | [🤝 贡献指南](CONTRIBUTING.md)

</div>

---

## 项目亮点

- **算子融合 + 正确性保证** — 每个 kernel 都附带 CPU 可测的 NumPy 参考实现，而非空口宣称加速。
- **生产级 FP8 GEMM 管线** — 显式 scale 管理与溢出处理，不是玩具级量化示例。
- **延迟驱动的自动调优 + 持久化缓存** — `TritonAutoTuner` + `ConfigCache`，不是一次性 benchmark 脚本。
- **OpenSpec 驱动开发** — 每个非平凡变更都先写设计文档再写代码，而非 YOLO 编码。

## 架构

```
User API (triton_ops.__init__)
    ├── 验证层 (device, dtype, shape, contiguity)
    ├── 计算参考层 (NumPy, CPU 可测)
    ├── 算子层 (Triton, GPU)
    └── 工具层 (autotuner, benchmark, performance metrics)
```

详见 [架构设计](https://lessup.github.io/triton-fused-ops/zh/internals/architecture) 与 [算子设计](https://lessup.github.io/triton-fused-ops/zh/internals/kernel-design) 文档。

## 快速开始

```bash
git clone https://github.com/LessUp/triton-fused-ops.git
cd triton-fused-ops
pip install -e ".[dev]"
```

**仅 CPU 验证**（无需 GPU）：

```bash
ruff format --check . && ruff check . && mypy triton_ops/
pytest tests/ -v -k "not cuda and not gpu" --ignore=tests/benchmarks/
python3 -m build
```

**完整 GPU 基准测试**（需 CUDA）：

```python
import torch
from triton_ops import fused_rmsnorm_rope, BenchmarkSuite
from triton_ops.kernels.rmsnorm_rope import fused_rmsnorm_rope_reference

x = torch.randn(2, 2048, 4096, device="cuda", dtype=torch.float16)
suite = BenchmarkSuite(warmup_runs=10, benchmark_runs=100)
result = suite.benchmark_kernel(
    fused_rmsnorm_rope, fused_rmsnorm_rope_reference,
    "fused_rmsnorm_rope", (2, 2048, 4096), x, ...
)
print(result.metrics.latency_ms)
```

## 性能

NVIDIA A100 SXM4 80GB 上的代表性数据（CUDA 12.1, PyTorch 2.1, Triton 2.1）。测量方法：10 轮 warmup + 100 轮 benchmark，计时区域前后使用 `torch.cuda.synchronize()` 同步。

| 算子 | 相比 PyTorch 加速 | 内存流量削减 |
|:--|:--:|:--:|
| `fused_rmsnorm_rope` | 最高 ~3.0&times; | ~40% |
| `fused_gated_mlp` | ~1.3x&ndash;1.8&times; | ~25% |
| `fp8_gemm` | ~1.2x&ndash;1.5&times; | ~50% (weights) |

详见 [性能可视化](https://lessup.github.io/triton-fused-ops/zh/guides/benchmark-visualization) 交互图表。

## 文档索引

| 章节 | 适合读者 | 核心收获 |
|:--|:--|:--|
| [开始使用](https://lessup.github.io/triton-fused-ops/zh/getting-started/) | 初次接触 | 5 分钟跑通第一个 kernel |
| [算子设计](https://lessup.github.io/triton-fused-ops/zh/internals/kernel-design) | 面试准备 | 融合思路、tiling 策略、内存优化 |
| [性能优化](https://lessup.github.io/triton-fused-ops/zh/guides/performance) | 调优实践 | 正确的测速方法、瓶颈分析 |
| [参考文献](https://lessup.github.io/triton-fused-ops/zh/references/) | 深度学习研究者 | 论文、竞品、技术栈全景 |

## 开发

本仓库的非平凡工作采用 **OpenSpec 驱动**。详见 [`AGENTS.md`](AGENTS.md)、[`CLAUDE.md`](CLAUDE.md) 和 [`openspec/README.md`](openspec/README.md)。

## 引用

```bibtex
@software{triton_fused_ops,
  title = {Triton Fused Ops: High-Performance GPU Kernels for Transformer Inference},
  author = {LessUp},
  year = {2025},
  url = {https://github.com/LessUp/triton-fused-ops},
  note = {Built on OpenAI Triton, PyTorch, and CUDA}
}
```

## 致谢

- [OpenAI Triton](https://github.com/triton-lang/triton) 提供编译器与 Python DSL
- [PyTorch](https://github.com/pytorch/pytorch) 提供张量运行时
- [NVIDIA](https://developer.nvidia.com/) 提供 CUDA、FP8 硬件与性能工具链

## 许可证

MIT.
