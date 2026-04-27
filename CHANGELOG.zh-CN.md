# 更新日志

本项目的所有显著变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/spec/v2.0.0.html)。

---

## [1.0.1] - 2026-04-27

### 修复
- FP8 GEMM 缩放验证：当输入已是 FP8 时现在要求缩放因子
- `fp8_gemm.py`、`fp8_quantize.py` 和 `api.py` 中的可选类型注解
- 类型语法一致性：统一使用 `typing` 模块的 `Tuple`
- `tuner.py` 异常处理：从 `Exception` 收窄为 `RuntimeError, OSError`
- `cache.py` 中的 TOCTOU 竞态条件：移除冗余的 `exists()` 检查

### 变更
- 归档已完成的 OpenSpec 变更 `prepare-project-for-archive`
- 移除 `_bmad/` 和 `_bmad-output/` 残留目录
- 从 `.gitignore` 中移除 `_bmad` 条目
- 配置 Git hooks 路径为 `.githooks`

### 新增
- pytest-cov 配置，覆盖率阈值 70%

---

## [未发布]

---

## [1.0.0] - 2026-04-16

### 🎉 首个稳定版本发布

我们很高兴宣布 Triton Fused Ops 的首个稳定版本！

### ✨ 发布亮点

- **生产就绪**: 所有算子经过充分测试和优化
- **双语文档**: 完整的中英文文档支持
- **专业日志**: 采用 Keep a Changelog 标准格式
- **全面测试**: 完整的测试套件，包含 Hypothesis 属性测试

### 🚀 核心功能

#### 融合算子

| 算子 | 加速比 | 内存节省 | 状态 |
|:-------|:-------:|:--------------:|:-------|
| **RMSNorm + RoPE 融合** | ~3 倍 | HBM 写入减少 50% | ✅ 稳定 |
| **Gated MLP 融合** | ~1.5 倍 | 减少 1 个中间张量 | ✅ 稳定 |
| **FP8 GEMM** | ~1.4 倍 | 50% 权重存储 | ✅ 稳定 |

#### 基础设施

- **自动调优框架**: 自动算子配置优化
  - 可配置搜索空间
  - 持久化结果缓存
  - 多种调优策略
  - 所有算子的预定义配置空间

- **基准测试套件**: 全面的性能测试
  - 与 PyTorch 参考实现的正确性验证
  - 带同步的性能测量
  - 指标报告生成

- **测试套件**: 全面的测试基础设施
  - 所有算子的单元测试
  - 使用 Hypothesis 的属性测试
  - 边界情况覆盖
  - CI/CD 集成

### 📚 文档

- 完整的 API 文档，包含示例
- 集成、性能优化、FP8 最佳实践的用户指南
- 内部架构和算子设计文档
- 双语支持（英文/中文）

### ⚠️ 破坏性变更

无 - 这是首个稳定版本。

### 📦 安装

```bash
pip install triton-fused-ops
```

### 🙏 致谢

感谢所有贡献者以及 OpenAI Triton 团队的出色工作。

---

## [0.2.0] - 2026-03-09

### 新增

#### 新功能
- **SwiGLU 正确性修复**: 修正了 Gated MLP 中的激活函数应用顺序，遵循标准 SwiGLU 公式：`output = activation(gate_proj(x)) * up_proj(x)`
- **FP8Linear 权重转置缓存**: 在 `FP8Linear` 中预转置并缓存权重，避免每次前向传递都调用 `.t().contiguous()`
- **改进的输入验证**: 添加了全面的验证和有用的错误信息

#### 基础设施
- GitHub Actions CI/CD 流水线，支持自动化测试
- GitHub Pages 文档站点
- 包含 Hypothesis 属性测试的全面测试套件

### 变更

- 重构算子启动模式以获得更好的错误处理
- 改进融合算子的内存访问模式
- 更新最低版本要求（PyTorch 2.0+，Triton 2.1+）

### 修复

- **RMSNorm batch_idx 计算**: 修复了融合 RMSNorm + RoPE 算子中错误的批次索引计算
- **FP8 GEMM 块大小**: 用自适应启发式替换硬编码值
- 长时间运行的自动调优会话中的内存泄漏

---

## [0.1.0] - 2024-01-01

### 新增

#### 融合算子
- **RMSNorm + RoPE 融合**: 融合 RMS 归一化和旋转位置编码的算子
  - 支持函数式和模块 API
  - 用于数值稳定性的可调 epsilon
  - 优化的内存访问模式
  - 相比分离操作实现 ~3 倍加速

- **Gated MLP 融合**: 用于门控 MLP 层的融合算子
  - 支持 SiLU (SwiGLU) 和 GELU (GeGLU) 激活函数
  - 门控和 Up 投影的单通道计算
  - 降低内存带宽需求
  - 相比标准实现 ~1.5 倍加速

- **FP8 GEMM**: FP8 量化矩阵乘法
  - E4M3 格式支持
  - 自动缩放计算
  - FP32 累加确保数值稳定性
  - 50% 内存节省，~1.4 倍加速

- **FP8 量化**: FP8 量化工具
  - 动态范围计算
  - 缩放因子计算
  - 带自动重试的溢出处理
  - 反量化支持

#### 基础设施
- **自动调优框架**: 自动算子配置优化
- **基准测试套件**: 全面的性能测试
- **测试套件**: 全面的测试基础设施

#### 文档
- 包含安装和使用说明的 README
- 包含示例的 API 文档
- 贡献指南
- 行为准则
- 中文文档 (README.zh-CN.md)

### 技术详情

| 要求 | 版本 |
|:------------|:--------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.0 |
| Triton | ≥ 2.1 |
| CUDA | ≥ 11.8 |
| GPU 架构 | Ampere (SM80+) |

---

## 版本历史摘要

| 版本 | 日期 | 亮点 |
|:--------|:-----|:-----------|
| 1.0.0 | 2026-04-16 | 首个稳定版本，双语文档 |
| 0.2.0 | 2026-03-09 | SwiGLU 修复，FP8Linear 优化，CI/CD |
| 0.1.0 | 2024-01-01 | 初始发布，包含所有核心算子 |

---

[未发布]: https://github.com/LessUp/triton-fused-ops/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/LessUp/triton-fused-ops/releases/tag/v1.0.0
[0.2.0]: https://github.com/LessUp/triton-fused-ops/releases/tag/v0.2.0
[0.1.0]: https://github.com/LessUp/triton-fused-ops/releases/tag/v0.1.0
