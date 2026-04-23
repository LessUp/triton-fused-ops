---
layout: default
title: 中文文档
nav_order: 3
has_children: true
permalink: /docs/zh/
---

# Triton Fused Ops 中文文档

本部分提供安装、API、集成与实现说明。

## 从这里开始

| 文档 | 用途 |
|:--|:--|
| [安装指南](getting-started/installation) | 环境要求与安装步骤 |
| [快速开始](getting-started/quickstart) | 最小可运行示例 |
| [示例教程](getting-started/examples) | 端到端使用示例 |

## API 与内部实现

| 章节 | 用途 |
|:--|:--|
| [核心算子](api/kernels) | 融合 RMSNorm+RoPE、Gated MLP、FP8 GEMM |
| [量化说明](api/quantization) | FP8 量化行为与注意事项 |
| [自动调优](api/autotuner) | 调优策略与缓存机制 |
| [基准测试](api/benchmark) | 基准接口与输出说明 |
| [架构设计](internals/architecture) | 库级架构设计 |

## 运行边界提醒

- Triton kernel 执行需要 CUDA GPU。
- CPU-only 环境适合仓库基线检查和非 kernel 逻辑验证。

## 快速链接

- [项目首页](../../)
- [English Docs](../en/)
- [更新日志](../../CHANGELOG)
- [贡献指南](https://github.com/LessUp/triton-fused-ops/blob/main/CONTRIBUTING.md)
