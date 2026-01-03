# Implementation Plan: Triton Fused Operators Library

## Overview

本实现计划将设计文档转化为可执行的编码任务。采用增量开发方式，每个任务都建立在前一个任务的基础上，确保代码始终可运行和可测试。

## Tasks

- [x] 1. 项目结构和核心接口设置
  - [x] 1.1 创建项目目录结构和配置文件
    - 创建 `triton_ops/` 包目录
    - 创建 `pyproject.toml` 配置 Python 包
    - 创建 `tests/` 目录结构
    - 配置 pytest 和 hypothesis
    - _Requirements: 5.1, 5.2_

  - [x] 1.2 定义核心数据模型和类型
    - 实现 `TensorSpec`, `RMSNormRoPEInput`, `GatedMLPInput`, `FP8GEMMInput` 数据类
    - 实现 `KernelMetrics`, `TuningResult` 数据类
    - 实现 `FP8Format` 规格类
    - _Requirements: 3.4_

  - [x] 1.3 实现错误处理和验证工具
    - 实现自定义异常类：`TritonKernelError`, `ShapeMismatchError`, `UnsupportedDtypeError`, `NumericalOverflowError`
    - 实现输入验证函数：`validate_rmsnorm_rope_inputs`, `validate_gated_mlp_inputs`, `validate_fp8_gemm_inputs`
    - _Requirements: 1.7, 3.5_

- [x] 2. Checkpoint - 确保基础设施就绪
  - 确保所有测试通过，如有问题请询问用户

- [x] 3. 实现 RMSNorm + RoPE 融合算子
  - [x] 3.1 实现 RMSNorm Triton kernel
    - 编写 `rmsnorm_kernel` 计算 `y = x * rsqrt(mean(x^2) + eps) * weight`
    - 实现块级并行和寄存器复用
    - _Requirements: 1.1_

  - [x] 3.2 实现 RoPE Triton kernel
    - 编写 `rope_kernel` 计算旋转位置编码
    - 实现 `rotate_half` 辅助函数
    - _Requirements: 1.2_

  - [x] 3.3 实现融合 RMSNorm + RoPE kernel
    - 将 RMSNorm 和 RoPE 合并为单个 kernel
    - 确保中间结果保留在寄存器中，不写回 HBM
    - 支持可变序列长度（1-8192）和隐藏维度（2048, 4096, 8192）
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 3.4 编写 RMSNorm + RoPE 属性测试
    - **Property 1: RMSNorm + RoPE Mathematical Correctness**
    - **Property 3: Dimension Flexibility (RMSNorm + RoPE 部分)**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4**

  - [x] 3.5 编写 NaN/Inf 边界情况单元测试
    - 测试 NaN 输入的传播
    - 测试 Inf 输入的传播
    - _Requirements: 1.7_

- [x] 4. 实现 Gated MLP 融合算子
  - [x] 4.1 实现 SiLU 和 GELU 激活函数
    - 编写 `silu_kernel`: `silu(x) = x * sigmoid(x)`
    - 编写 `gelu_kernel`: `gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))`
    - _Requirements: 2.2, 2.3_

  - [x] 4.2 实现融合 Gated MLP kernel
    - 编写 `fused_gated_mlp_kernel` 计算 `output = gate_proj(x) * activation(up_proj(x))`
    - 支持 SiLU 和 GELU 激活函数切换
    - 支持批量大小（1-64）和中间维度（5632, 11264, 22528）
    - _Requirements: 2.1, 2.4, 2.5, 2.6_

  - [x] 4.3 编写 Gated MLP 属性测试
    - **Property 2: Gated MLP Correctness with Activation Functions**
    - **Property 3: Dimension Flexibility (Gated MLP 部分)**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

- [x] 5. Checkpoint - 确保融合算子正确
  - 确保所有测试通过，如有问题请询问用户

- [x] 6. 实现 FP8 量化和 GEMM
  - [x] 6.1 实现 FP8 量化/反量化 kernel
    - 编写 `quantize_fp8_kernel` 将 FP16/BF16 转换为 FP8 E4M3
    - 编写 `dequantize_fp8_kernel` 将 FP8 转换回 FP16/BF16
    - 实现动态缩放因子计算
    - 实现溢出检测和处理
    - _Requirements: 3.4, 3.5_

  - [x] 6.2 编写 FP8 量化往返属性测试
    - **Property 5: FP8 Quantization Round-Trip**
    - **Validates: Requirements 3.4**

  - [x] 6.3 实现 FP8 GEMM kernel
    - 编写 `fp8_gemm_kernel` 使用 FP8 输入进行矩阵乘法
    - 使用 FP32 累加器保证数值稳定性
    - 输出 FP16/BF16 格式
    - 使用 Block Pointer 优化内存访问
    - _Requirements: 3.1, 3.2, 3.3, 3.6_

  - [x] 6.4 编写 FP8 GEMM 属性测试
    - **Property 4: FP8 GEMM Correctness**
    - **Property 6: FP8 Accuracy vs FP16 Baseline**
    - **Validates: Requirements 3.1, 3.8**

- [x] 7. 实现 Auto-Tuning 框架
  - [x] 7.1 实现配置空间定义
    - 定义 `RMSNORM_ROPE_CONFIGS`, `GATED_MLP_CONFIGS`, `FP8_GEMM_CONFIGS`
    - 实现配置生成器
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 7.2 实现 TritonAutoTuner 类
    - 实现 `tune()` 方法进行配置搜索
    - 实现 warmup 运行逻辑
    - 实现性能指标收集（latency, throughput, bandwidth）
    - _Requirements: 4.4, 4.6_

  - [x] 7.3 实现配置缓存机制
    - 实现 `get_cached_config()` 方法
    - 实现缓存存储和加载
    - _Requirements: 4.5_

  - [x] 7.4 编写 Auto-Tuner 属性测试
    - **Property 7: Auto-Tuner Cache Consistency**
    - **Validates: Requirements 4.5**

- [x] 8. Checkpoint - 确保核心功能完成
  - 确保所有测试通过，如有问题请询问用户

- [x] 9. 实现基准测试套件
  - [x] 9.1 实现基准测试框架
    - 创建 `BenchmarkSuite` 类
    - 实现 PyTorch 原生操作对比
    - 实现 cuBLAS/cuDNN 基线对比（如可用）
    - _Requirements: 5.1, 5.2_

  - [x] 9.2 实现正确性验证
    - 实现数值精度验证函数
    - 实现容差检查逻辑
    - _Requirements: 5.3_

  - [x] 9.3 编写基准测试正确性属性测试
    - **Property 8: Benchmark Correctness Verification**
    - **Validates: Requirements 5.3**

  - [x] 9.4 实现性能报告生成
    - 实现 HBM 带宽利用率测量
    - 实现执行时间测量
    - 生成人类可读的性能报告
    - _Requirements: 5.4, 5.5, 5.6_

- [x] 10. 实现 Python 接口封装
  - [x] 10.1 创建 torch.nn.Module 封装
    - 实现 `FusedRMSNormRoPE` Module
    - 实现 `FusedGatedMLP` Module
    - 实现 `FP8Linear` Module
    - 支持 autograd 反向传播（如需要）
    - _Requirements: 1.1-1.5, 2.1-2.6, 3.1-3.8_

  - [x] 10.2 创建便捷 API
    - 实现函数式接口：`fused_rmsnorm_rope()`, `fused_gated_mlp()`, `fp8_gemm()`
    - 添加文档字符串和类型注解
    - _Requirements: 1.1-1.5, 2.1-2.6, 3.1-3.8_

- [x] 11. Final Checkpoint - 确保所有测试通过
  - 运行完整测试套件
  - 确保所有属性测试通过
  - 如有问题请询问用户

## Notes

- 每个任务都引用了具体的需求以便追溯
- Checkpoint 任务确保增量验证
- 属性测试验证通用正确性属性
- 单元测试验证特定示例和边界情况
- 所有任务都是必须完成的，包括测试任务
