---
layout: default
title: 架构设计
parent: 内部实现
grand_parent: 中文文档
nav_order: 1
description: "仓库的模块结构与各层职责关系"
---

# 架构设计

仓库整体围绕一个小而清晰的公开 API 层构建，下面连接输入校验、Triton kernel 实现、性能工具层和共享数据模型。

## 模块地图

```text
triton_ops/
├── __init__.py          # 根包导出
├── api.py               # 便捷 API 包装
├── models.py            # dataclass 与指标/结果容器
├── exceptions.py        # 自定义异常类型
├── validation.py        # 运行时输入校验
├── utils.py             # 公共 helper 与常量
├── kernels/
│   ├── rmsnorm_rope.py
│   ├── gated_mlp.py
│   ├── fp8_gemm.py
│   └── fp8_quantize.py
├── autotuner/
│   ├── configs.py
│   ├── tuner.py
│   └── cache.py
└── benchmark/
    ├── correctness.py
    ├── report.py
    └── suite.py
```

## 职责拆分

### 公开 API 层

`triton_ops.__init__` 是主要用户入口，导出 kernel、模块封装、量化 helper、benchmark 类、自动调优工具、dataclass 和异常类型。

`triton_ops.api` 也提供了一层便捷封装，但根包仍然是主要公开接口。

### 校验层

`validation.py` 统一管理输入契约：

- device 放置，
- dtype 支持，
- contiguous 要求，
- shape 兼容性，
- 标量参数检查。

这样 kernel 入口就不用重复写大量样板校验逻辑，测试和 wrapper 也能复用同一套规则。

### Kernel 层

`kernels/` 中放的是 Triton 实现，以及用于正确性对照的 PyTorch reference 实现。

每个 kernel 模块通常包含：

- Triton kernel 本体，
- 用户可调用的 Python launcher，
- reference 函数，
- 可选的 `nn.Module` 封装。

### 支撑工具层

自动调优和 benchmark 都独立于 kernel 运行路径存在。它们的目标是帮助实验、验证和报告，而不是把所有调优逻辑都隐式塞进每次 API 调用里。

## 架构意图

这个结构偏向于：

- 显式的运行契约，
- 可验证的 reference 路径，
- 小而明确的导出原语集合，
- 可独立复用的支持代码。

## 关键边界

- 仓库不提供完整的 Transformer 模型栈。
- 这些融合算子更适合作为更大推理系统中的原语组件。
- benchmark 和 autotuning 是伴随工具层，而不是必须进入正常运行路径的中间层。
