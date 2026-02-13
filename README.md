# Triton Fused Operators Library

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Triton](https://img.shields.io/badge/Triton-2.1+-76B900?logo=nvidia&logoColor=white)

高性能 Triton 算子库，专为 Transformer 模型优化，支持算子融合和 FP8 量化。

## 特性

- **RMSNorm + RoPE 融合算子**: 将归一化和旋转位置编码融合为单个 kernel，减少 3 次 HBM 访问到 1 次
- **Gated MLP 融合算子**: 融合门控 MLP 的投影和激活计算，支持 SiLU 和 GELU
- **FP8 量化 GEMM**: 8-bit 浮点矩阵乘法，支持动态缩放，精度损失 < 1%
- **Auto-Tuning 框架**: 自动搜索最优的 BLOCK_SIZE、num_warps 等超参数
- **基准测试套件**: 对比 PyTorch/cuBLAS 基线，验证正确性和性能

## 安装

```bash
pip install -e ".[dev]"
```

### 依赖

- Python >= 3.9
- PyTorch >= 2.0
- Triton >= 2.1
- CUDA >= 11.8

## 快速开始

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
gate_weight = torch.randn(11264, 4096, device='cuda', dtype=torch.float16)
up_weight = torch.randn(11264, 4096, device='cuda', dtype=torch.float16)
output = fused_gated_mlp(x, gate_weight, up_weight, activation='silu')

# FP8 GEMM（自动量化）
a = torch.randn(1024, 4096, device='cuda', dtype=torch.float16)
b = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
output = fp8_gemm(a, b)
```

### Module API

```python
import torch
from triton_ops import FusedRMSNormRoPE, FusedGatedMLP, FP8Linear

# 作为 nn.Module 使用
class TransformerBlock(torch.nn.Module):
    def __init__(self, hidden_dim=4096, head_dim=64, intermediate_dim=11264):
        super().__init__()
        self.norm = FusedRMSNormRoPE(hidden_dim, head_dim)
        self.mlp = FusedGatedMLP(hidden_dim, intermediate_dim, activation='silu')
        self.proj = FP8Linear(intermediate_dim, hidden_dim)
    
    def forward(self, x, cos, sin):
        x = self.norm(x, cos, sin)
        x = self.mlp(x)
        x = self.proj(x)
        return x
```

### FP8 量化

```python
from triton_ops import quantize_fp8, dequantize_fp8

# 量化
tensor = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)
quantized, scale = quantize_fp8(tensor)

# 反量化
recovered = dequantize_fp8(quantized, scale)
```

### Auto-Tuning

```python
from triton_ops import TritonAutoTuner, RMSNORM_ROPE_CONFIGS

# 创建 tuner
tuner = TritonAutoTuner(
    kernel_fn=my_kernel,
    config_space=RMSNORM_ROPE_CONFIGS,
    warmup_runs=10,
    benchmark_runs=100,
)

# 运行调优
result = tuner.tune(
    *args,
    problem_size=(batch, seq_len, hidden_dim),
    device="cuda:0",
)
print(f"Best config: {result.best_config}")
print(f"Latency: {result.metrics.latency_ms:.3f} ms")
```

### 基准测试

```python
from triton_ops import BenchmarkSuite

suite = BenchmarkSuite(warmup_runs=10, benchmark_runs=100)

# 运行基准测试
results = suite.benchmark_rmsnorm_rope(
    batch_sizes=[1, 4, 8],
    seq_lens=[128, 512, 2048],
    hidden_dims=[2048, 4096],
)

# 生成报告
report = suite.generate_report()
print(report)
suite.save_report("benchmark_report.txt")
```

## 运行测试

```bash
# 运行所有测试（需要 CUDA）
pytest tests/ -v

# 运行属性测试（更多迭代，用于 CI）
pytest tests/ -v --hypothesis-profile=ci

# 运行特定测试
pytest tests/test_rmsnorm_rope.py -v
pytest tests/test_fp8_gemm.py -v
```

## 运行基准测试

```bash
# RMSNorm + RoPE 基准测试
python -m tests.benchmarks.bench_rmsnorm_rope

# Gated MLP 基准测试
python -m tests.benchmarks.bench_gated_mlp

# FP8 GEMM 基准测试
python -m tests.benchmarks.bench_fp8_gemm
```

## 项目结构

```
triton_ops/
├── __init__.py         # 主入口，导出所有 API
├── api.py              # 便捷 API 封装
├── models.py           # 数据模型（TensorSpec, KernelMetrics 等）
├── exceptions.py       # 自定义异常
├── validation.py       # 输入验证
├── kernels/
│   ├── rmsnorm_rope.py # RMSNorm + RoPE 融合算子
│   ├── gated_mlp.py    # Gated MLP 融合算子
│   ├── fp8_gemm.py     # FP8 GEMM
│   └── fp8_quantize.py # FP8 量化/反量化
├── autotuner/
│   ├── tuner.py        # Auto-tuning 框架
│   ├── configs.py      # 配置空间定义
│   └── cache.py        # 配置缓存
└── benchmark/
    ├── suite.py        # 基准测试套件
    ├── correctness.py  # 正确性验证
    └── report.py       # 性能报告生成
```

## 性能亮点

- **RMSNorm + RoPE**: 通过算子融合减少 3 次 HBM 访问到 1 次，带宽利用率可达 90%+
- **FP8 GEMM**: 相比 FP16 减少 50% 显存占用，精度损失 < 1%
- **Auto-Tuning**: 自动搜索最优配置，适配不同硬件和问题规模

## 面试亮点

> "针对 Transformer 的解码阶段，我用 Triton 实现了一个融合算子，减少了 3 次显存读写（HBM Access），使得该层的带宽利用率（Memory Bandwidth Utilization）达到了 90% 以上。"

> "我处理了 FP8 量化中的精度溢出问题，通过在 Triton 中动态缩放 scaling factor 实现了与 FP16 几乎无损的精度。"

## 许可证

MIT License
