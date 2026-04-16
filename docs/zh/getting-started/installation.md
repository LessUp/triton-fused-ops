---
layout: default
title: "安装指南 — Triton Fused Ops"
description: "Triton Fused Ops 安装指南 - 系统要求和安装说明"
---

# 📦 安装指南

本指南介绍 Triton Fused Ops 及其依赖项的安装方法。

---

## ✅ 系统要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|:----------|:--------|:------------|
| **GPU** | NVIDIA Ampere (SM80) | NVIDIA H100、A100 或 RTX 4090 |
| **显存** | 8 GB | 16 GB+（大模型需要） |
| **CUDA** | 11.8 | 12.1+ |

### 软件要求

| 依赖项 | 版本 | 说明 |
|:-----------|:--------|:------|
| **Python** | ≥ 3.9 | 推荐 Python 3.10 或 3.11 |
| **PyTorch** | ≥ 2.0.0 | 需要 CUDA 支持 |
| **Triton** | ≥ 2.1.0 | OpenAI Triton |
| **NumPy** | ≥ 1.21.0 | 张量操作所需 |

---

## 🚀 安装方法

### 方法 1：开发安装（推荐）

获取最新功能和开发版本：

```bash
# 克隆仓库
git clone https://github.com/LessUp/triton-fused-ops.git
cd triton-fused-ops

# 以可编辑模式安装，包含开发依赖
pip install -e ".[dev]"
```

### 方法 2：仅核心安装

生产环境使用，依赖最小化：

```bash
pip install -e .
```

### 方法 3：使用 uv（快速）

如果您使用 `uv` 进行包管理：

```bash
# 包含开发依赖
uv pip install -e ".[dev]"

# 仅核心
uv pip install -e .
```

---

## 🔧 验证安装

### 检查 CUDA 可用性

```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 测试 Triton Fused Ops

```python
import torch
from triton_ops import fused_rmsnorm_rope

# 测试基本功能
x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
weight = torch.ones(4096, device='cuda', dtype=torch.float16)
cos = torch.randn(128, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(128, 64, device='cuda', dtype=torch.float16)

output = fused_rmsnorm_rope(x, weight, cos, sin)
print(f"✅ 输出形状: {output.shape}")
print(f"✅ 输出类型: {output.dtype}")
```

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行覆盖率测试
pytest tests/ -v --cov=triton_ops

# 运行特定测试文件
pytest tests/test_fp8_gemm.py -v
```

---

## 🐛 故障排除

### 问题："CUDA is not available"

**原因：** PyTorch 未安装 CUDA 支持，或 CUDA 配置不正确。

**解决方法：**
```bash
# 重新安装带 CUDA 支持的 PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 问题："Triton not found"

**原因：** 未安装 Triton 包。

**解决方法：**
```bash
pip install triton>=2.1.0
```

### 问题：编译扩展导入错误

**原因：** PyTorch 和 Triton 版本不匹配。

**解决方法：**
```bash
# 升级两者到最新兼容版本
pip install --upgrade torch triton
```

---

## 📋 GPU 架构支持

| 架构 | FP16 | FP8 | 说明 |
|:-------------|:----:|:---:|:------|
| Ampere (A100) | ✅ | ⚠️ 模拟 | 生产就绪 |
| Ada (RTX 4090) | ✅ | ✅ | 边缘部署 |
| Hopper (H100) | ✅ | ✅ | FP8 最佳 |

---

## 🎯 下一步

- [快速开始指南](./quickstart.md) — 运行您的第一个融合算子
- [示例教程](./examples.md) — 从实用示例学习
- [API 参考](../api/kernels.md) — 探索 API

---

<div align="center">

**[⬆ 返回顶部](#-安装指南)** | **[← 返回文档](../)**

</div>
