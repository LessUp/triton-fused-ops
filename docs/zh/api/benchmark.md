---
title: 基准测试
description: "基准测试编排、正确性验证与报告生成"
---

# 基准测试

本仓库的 benchmark 层主要由类与 helper 组成，而不是一组根包级独立 benchmark 函数。

## `BenchmarkSuite`

```python
BenchmarkSuite(
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    rtol: float = 1e-3,
    atol: float = 1e-5,
)
```

主要方法：

- `benchmark_kernel(...)`
- `compare_with_pytorch(...)`
- `benchmark_rmsnorm_rope(...)`
- `benchmark_gated_mlp(...)`
- `benchmark_fp8_gemm(...)`
- `generate_report(format="text" | "json")`
- `save_report(filepath, format="text" | "json")`

## `CorrectnessVerifier`

```python
CorrectnessVerifier(rtol: float = 1e-3, atol: float = 1e-5)
```

常用方法：

- `verify(actual, expected) -> tuple[bool, dict]`
- `verify_allclose(actual, expected) -> bool`
- `compute_relative_error(actual, expected) -> float`

其中 `verify` 会返回更详细的统计信息，比如最大绝对误差、平均相对误差和违规元素数量。

## 独立正确性 helper

位于 `triton_ops.benchmark.correctness`：

- `verify_fp8_accuracy(fp8_result, fp16_baseline, max_relative_error=0.01)`
- `verify_nan_inf_propagation(output, input_has_nan, input_has_inf)`

如果你不想走 `BenchmarkSuite`，这些函数也适合单独做数值验证。

## 报告对象

`triton_ops.benchmark.report` 定义了：

- `BenchmarkResult`
- `ComparisonResult`
- `PerformanceReport`

`PerformanceReport` 支持：

- `generate_text_report()` 生成人类可读文本
- `generate_json_report()` 生成 JSON

## 重要提醒

基准测试模块中的专用 benchmark 方法是 GPU 导向的，因为它们会直接在 CUDA 上分配测试张量。若只是做仓库健康检查，应优先使用测试、lint、类型检查与构建命令，而不是 GPU benchmark。

## 相关指标 helper

吞吐量和带宽的计算 helper 见自动调优页：

- `compute_gemm_metrics`
- `compute_elementwise_metrics`
