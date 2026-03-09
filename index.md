---
layout: default
title: "Triton Fused Ops — 高性能 Triton 算子库"
description: "支持 RMSNorm+RoPE 融合 · Gated MLP 融合 · FP8 GEMM · Auto-Tuning，专为 Transformer 模型优化"
---

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/LessUp/triton-fused-ops/blob/main/LICENSE)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Triton](https://img.shields.io/badge/Triton-2.1+-76B900?logo=nvidia&logoColor=white)

高性能 Triton 算子库，专为 Transformer 模型优化。通过算子融合大幅减少 HBM 访问次数，结合 FP8 量化和自动调优，为大语言模型推理提供接近硬件极限的吞吐。

---

## 核心算子

| 算子 | 描述 | 关键优化 |
|------|------|----------|
| **RMSNorm + RoPE** | 归一化 + 旋转位置编码融合为单个 kernel | HBM 访问 3→1 次，带宽利用率 90%+ |
| **Gated MLP** | 门控投影 + 激活（SiLU / GELU）单 pass 融合 | 减少中间张量显存分配 |
| **FP8 GEMM** | 8-bit 浮点矩阵乘法，动态缩放 | 显存占用 −50%，精度损失 < 1% |
| **FP8 量化** | 动态范围 + Scale Factor 计算 | 支持 E4M3 / E5M2 格式 |

## 性能亮点

- **RMSNorm + RoPE 融合** — 将归一化、旋转位置编码、残差连接融合为一次 kernel launch，Memory Bandwidth Utilization 达 **90%+**
- **FP8 GEMM** — 相比 FP16 减少 **50%** 显存占用，通过动态 scaling factor 实现与 FP16 几乎无损精度
- **Auto-Tuning** — 自动搜索最优 `BLOCK_SIZE`、`num_warps`、`num_stages` 等超参数，适配不同 GPU 架构和问题规模

## 快速开始

```bash
pip install -e ".[dev]"
```

### 依赖要求

| 依赖 | 最低版本 |
|------|---------|
| Python | 3.9 |
| PyTorch | 2.0 |
| Triton | 2.1 |
| CUDA | 11.8 |

## 使用示例

### 函数式 API

```python
import torch
from triton_ops import fused_rmsnorm_rope, fused_gated_mlp, fp8_gemm

# RMSNorm + RoPE 融合
x = torch.randn(2, 1024, 4096, device='cuda', dtype=torch.float16)
weight = torch.randn(4096, device='cuda', dtype=torch.float16)
cos = torch.randn(1024, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(1024, 64, device='cuda', dtype=torch.float16)
output = fused_rmsnorm_rope(x, weight, cos, sin)

# Gated MLP 融合
x = torch.randn(2, 1024, 4096, device='cuda', dtype=torch.float16)
gate_w = torch.randn(11264, 4096, device='cuda', dtype=torch.float16)
up_w = torch.randn(11264, 4096, device='cuda', dtype=torch.float16)
output = fused_gated_mlp(x, gate_w, up_w, activation='silu')

# FP8 GEMM（自动量化）
a = torch.randn(1024, 4096, device='cuda', dtype=torch.float16)
b = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
output = fp8_gemm(a, b)
```

### Module API — 嵌入 Transformer Block

```python
import torch
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP, FP8Linear

class TransformerBlock(torch.nn.Module):
    def __init__(self, hidden=4096, head=64, inter=11264):
        super().__init__()
        self.norm = FusedRMSNormRoPE(hidden, head)
        self.mlp = FusedGatedMLP(hidden, inter, activation='silu')
        self.proj = FP8Linear(inter, hidden)

    def forward(self, x, cos, sin):
        x = self.norm(x, cos, sin)
        x = self.mlp(x)
        return self.proj(x)
```

### Auto-Tuning

```python
from triton_ops import TritonAutoTuner, RMSNORM_ROPE_CONFIGS

tuner = TritonAutoTuner(
    kernel_fn=my_kernel,
    config_space=RMSNORM_ROPE_CONFIGS,
    warmup_runs=10, benchmark_runs=100,
)
result = tuner.tune(*args, problem_size=(batch, seq_len, hidden_dim))
print(f"Best: {result.best_config}  Latency: {result.metrics.latency_ms:.3f} ms")
```

## 项目结构

```
triton_ops/
├── __init__.py          # 主入口
├── api.py               # 便捷 API 封装
├── models.py            # TensorSpec, KernelMetrics 等
├── validation.py        # 输入验证
├── kernels/
│   ├── rmsnorm_rope.py  # RMSNorm + RoPE 融合
│   ├── gated_mlp.py     # Gated MLP 融合
│   ├── fp8_gemm.py      # FP8 GEMM
│   └── fp8_quantize.py  # FP8 量化/反量化
├── autotuner/
│   ├── tuner.py         # Auto-tuning 框架
│   ├── configs.py       # 配置空间
│   └── cache.py         # 配置缓存
└── benchmark/
    ├── suite.py          # 基准测试套件
    ├── correctness.py    # 正确性验证
    └── report.py         # 性能报告
```

## 运行测试与基准

```bash
# 全量测试（需要 CUDA）
pytest tests/ -v

# 基准测试
python -m tests.benchmarks.bench_rmsnorm_rope
python -m tests.benchmarks.bench_gated_mlp
python -m tests.benchmarks.bench_fp8_gemm
```

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.9+, Triton DSL |
| 框架 | PyTorch 2.0+ |
| GPU | CUDA 11.8+（Ampere / Ada / Hopper） |
| 测试 | pytest, Hypothesis (property-based) |
| 质量 | Ruff, EditorConfig |
| CI/CD | GitHub Actions |

---

## 最近更新

| 日期 | 变更 |
|------|------|
| 2026-03-10 | GitHub Pages 优化 — SEO 元数据、kramdown GFM、sparse checkout、changelog 索引 |
| 2026-03-09 | **v0.2.0 重大重构** — SwiGLU 正确性修复、FP8Linear 权重转置缓存、RMSNorm batch_idx 修复 |

[查看完整更新日志 →](changelog/)

---

## 链接

- [完整 README](README.md)
- [更新日志](changelog/) · [CHANGELOG.md](CHANGELOG.md)
- [贡献指南](CONTRIBUTING.md)
- [行为准则](CODE_OF_CONDUCT.md)
