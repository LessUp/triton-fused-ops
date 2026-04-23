# Triton Fused Operators Library

> **Status:** ✅ Archived
> **Created:** 2026-03-09
> **Archived:** 2026-04-23

---

## Why

Transformer 模型的解码阶段存在显著的内存带宽瓶颈。每次解码步骤需要多次访问 HBM（High Bandwidth Memory），导致延迟增加和吞吐量下降。具体问题包括：

1. **RMSNorm 和 RoPE 分离执行**：两个操作各自读写 HBM，造成冗余内存访问
2. **Gated MLP 激活函数未融合**：gate projection 和 up projection 的中间结果写回内存
3. **FP16/BF16 精度限制**：无法充分利用 FP8 量化带来的内存和计算优势

## What Changes

1. **实现 RMSNorm + RoPE 融合算子**：将归一化和位置编码合并为单个 kernel，减少 HBM 访问
2. **实现 Gated MLP 融合算子**：融合门控投影、上投影和激活函数
3. **实现 FP8 量化 GEMM**：使用 FP8 E4M3 格式进行矩阵乘法，减少显存占用
4. **实现 Auto-Tuning 框架**：自动搜索最优 kernel 配置

## Capabilities

### New Capabilities

| Capability | Description | Status |
|------------|-------------|--------|
| `fused_rmsnorm_rope` | RMSNorm + RoPE fusion kernel | ✅ Implemented |
| `fused_gated_mlp` | Gated MLP with SiLU/GELU fusion | ✅ Implemented |
| `fp8_gemm` | FP8 matrix multiplication | ✅ Implemented |
| `fp8_quantize` | FP8 quantization utilities | ✅ Implemented |
| `auto_tuner` | Kernel configuration search | ✅ Implemented |
| `benchmark_suite` | Performance validation tools | ✅ Implemented |

### Modified Capabilities

None (initial implementation)

## Impact

### Files Added

```
triton_ops/
├── kernels/
│   ├── rmsnorm_rope.py      # Fused RMSNorm + RoPE
│   ├── gated_mlp.py          # Fused Gated MLP
│   └── fp8_gemm.py           # FP8 GEMM operations
├── autotuner/
│   ├── __init__.py
│   ├── config.py             # Configuration spaces
│   └── tuner.py              # Tuning logic
├── benchmark/
│   ├── __init__.py
│   └── suite.py              # Benchmark framework
├── models.py                 # Data models
├── exceptions.py             # Custom exceptions
├── validation.py             # Input validation
└── api.py                    # Public API
```

### Performance Impact

| Operator | Baseline | Optimized | Improvement |
|----------|----------|-----------|-------------|
| RMSNorm + RoPE | 3 HBM accesses | 1 HBM access | ~66% reduction |
| Gated MLP | 3 kernel launches | 1 kernel launch | ~67% reduction |
| FP8 GEMM | FP16 memory | FP8 memory | ~50% memory reduction |
