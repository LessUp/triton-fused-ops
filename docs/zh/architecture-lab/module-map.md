---
title: 模块地图
description: triton_ops 如何被拆分为公共接口、kernels、reference 与支撑工具
---

# 模块地图

这个仓库足够紧凑，理论上可以装进一个心智模型里；前提是每个模块的职责必须一直保持显式。

## 顶层地图

| 路径 | 角色 |
| :-- | :-- |
| `triton_ops/__init__.py` | 根级公共导出，也是用户最稳定的导入面 |
| `triton_ops/kernels/` | Triton launcher、模块封装以及私有实现 kernel |
| `triton_ops/reference/` | 用于正确性校验与 CPU 测试的 reference 实现 |
| `triton_ops/benchmark/` | benchmark 编排、correctness 校验与报告生成 |
| `triton_ops/autotuner/` | 配置搜索、预设配置集与缓存持久化 |
| `triton_ops/performance.py` | 用 latency + shape 上下文推导 Performance metrics |
| `triton_ops/models.py` | 共享 dataclass 与结果容器 |
| `triton_ops/validation.py` | 运行时契约与声明式验证对象 |
| `triton_ops/exceptions.py` | 类型化失败模式 |

## 架构意图

### `triton_ops/__init__.py`

顶层包有意聚合稳定公共表面，让文档、测试与下游代码围绕同一组名称交流。

### `triton_ops/kernels/`

这里保存每个 Kernel family 的实现型 launch 路径。私有 `*_kernel` Triton 函数都在这里，但它们不是公共契约的一部分。

### `triton_ops/reference/`

reference 实现不是附属品，而是 correctness 检查与 CPU-only 回归覆盖的证明接缝。

### `triton_ops/benchmark/` 与 `triton_ops/autotuner/`

两者是并列子系统，而不是隐藏运行时层。Benchmarking 负责报告证据，Auto-Tuning 负责搜索配置，这种分离让测量语言保持诚实。

## 依赖阅读启发式

按“由外向内”的顺序理解依赖：

1. 公共导入面；
2. validation 与 models；
3. kernels 与 reference；
4. benchmark 与 autotuner 支撑层。

这样的顺序最利于维护，也更适合作为代码审查或面试中的解释路径。
