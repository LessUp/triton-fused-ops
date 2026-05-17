---
title: 导读
description: 用最高信噪比理解 Triton Fused Ops 的定位、术语与证据模型
---

# 导读

这一节的目标很简单：用最短路径说明仓库是什么、不是什麽，以及后续文档该怎么读，才不会把术语混用。

## 项目姿态

Triton Fused Ops 是一个面向 Transformer 推理的 kernel 库。仓库围绕少量 **Kernel family** 入口组织，同时提供 validation、reference、Benchmarking、Auto-Tuning 与 Performance metrics 这些支撑层。

它的取向不是“功能越多越好”，而是工业化地把几个热点路径做深做透：

- 只优化窄而关键的高频算子路径；
- 用 reference 实现维持可审计的正确性；
- 把运行时契约显式化，而不是散落在调用栈里；
- 让性能说法绑定测量方法，而不是脱离实验上下文。

## 核心词汇

| 术语 | 在本仓库里的含义 | 明确不表示什么 |
| :-- | :-- | :-- |
| **Kernel family** | 面向用户的融合操作族，如 `fused_rmsnorm_rope`、`fused_gated_mlp`、`fp8_gemm` | 任意内部 Triton kernel 的泛称 |
| **Benchmarking** | 验证正确性、测量延迟、产出报告的一组仓库工具 | Auto-Tuning 或泛化 profiling |
| **Auto-Tuning** | 针对 launch 参数做配置搜索，并缓存最低延迟结果 | 端到端 benchmark 或运行时魔法 |
| **Performance metrics** | 用延迟加问题形状推导出来的吞吐与带宽指标 | 只有原始 latency 的数字 |

## 仓库真正交付的层次

| 层次 | 主要代码路径 | 存在理由 |
| :-- | :-- | :-- |
| 公共 API surface | `triton_ops.__init__.py` | 给用户一个稳定、集中的导入入口 |
| validation 层 | `triton_ops.validation` | 在 launch 前拦截非法 device、dtype、shape 与 contiguous 组合 |
| kernel 层 | `triton_ops.kernels.*` | 执行优化后的 Triton 实现 |
| reference 层 | `triton_ops.reference.*` | 提供可 CPU 测试、可核对语义的基线实现 |
| 工具层 | `triton_ops.benchmark`、`triton_ops.autotuner`、`triton_ops.performance` | 测量、比较并解释 kernel 行为 |

## 证据模型

技术白皮书只有在读者能把每个结论追溯到具体机制时才有价值。

1. **能力主张** 应该能在导出的 API surface 上直接看到。
2. **正确性主张** 应该回到 `triton_ops.reference` 与 `CorrectnessVerifier`。
3. **运行时边界主张** 应该能在 `triton_ops.validation` 与异常类型里找到。
4. **性能主张** 必须区分 Benchmarking 与 Auto-Tuning；只有形状上下文明确时才谈 Performance metrics。

## 下一步读哪里

<div class="link-grid link-grid-3">
  <a class="info-card" href="../academy/">
    <span class="card-kicker">学院</span>
    <strong>先把系统层次看完整</strong>
    <span>从系统总览开始，再进入单个算子族。</span>
  </a>
  <a class="info-card" href="../kernel-families/">
    <span class="card-kicker">算子族</span>
    <strong>逐族比较工作负载与证据</strong>
    <span>对照 Fused RMSNorm + RoPE、Fused Gated MLP 与 FP8 栈的契约和评估方式。</span>
  </a>
  <a class="info-card" href="../architecture-lab/">
    <span class="card-kicker">架构实验室</span>
    <strong>看模块缝隙与运行边界</strong>
    <span>当你需要实现视角而不是产品叙事时，从这里进入。</span>
  </a>
</div>
