---
layout: default
title: "贡献指南 — Triton Fused Ops"
description: "Triton Fused Ops 项目贡献流程、代码规范与提交信息格式"
---

[← 返回首页]({{ site.baseurl }}/)

# Contributing

感谢你对本项目的关注！本文档介绍如何参与 Triton Fused Ops 的开发贡献。

## 目录

- [行为准则](#行为准则)
- [开发环境搭建](#开发环境搭建)
- [开发流程](#开发流程)
- [代码规范](#代码规范)
- [测试要求](#测试要求)
- [提交信息格式](#提交信息格式)
- [Pull Request 流程](#pull-request-流程)
- [文档贡献](#文档贡献)

---

## 行为准则

本项目采用 [Contributor Covenant](https://www.contributor-covenant.org/) 行为准则。参与本项目即表示你同意遵守其条款。详情请参阅 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

---

## 开发环境搭建

### 前置要求

| 依赖 | 最低版本 | 推荐版本 |
|------|---------|----------|
| Python | 3.9 | 3.10+ |
| PyTorch | 2.0 | 2.1+ |
| Triton | 2.1 | 最新版 |
| CUDA | 11.8 | 12.1+ |
| GPU | Ampere (SM80) | Hopper (SM90) |

### 安装步骤

```bash
# 1. Fork 并克隆仓库
git clone https://github.com/YOUR_USERNAME/triton-fused-ops.git
cd triton-fused-ops

# 2. 创建虚拟环境（推荐使用 conda）
conda create -n triton-ops python=3.10
conda activate triton-ops

# 3. 安装开发依赖
pip install -e ".[dev]"

# 4. 验证安装
python -c "import triton_ops; print(triton_ops.__version__)"
pytest tests/ -v --collect-only
```

### IDE 配置

推荐使用 VS Code 或 PyCharm，并配置：

- **Formatter**: Black (line-length: 100)
- **Linter**: Ruff
- **Type Checker**: MyPy

---

## 开发流程

### 分支命名规范

| 类型 | 命名格式 | 示例 |
|------|----------|------|
| 功能 | `feature/描述` | `feature/add-flash-attn` |
| 修复 | `fix/描述` | `fix/rmsnorm-overflow` |
| 性能 | `perf/描述` | `perf/optimize-gemm` |
| 文档 | `docs/描述` | `docs/api-reference` |
| 重构 | `refactor/描述` | `refactor/autotuner` |

### 工作流程

```
1. 从 main 创建特性分支
   git checkout -b feature/your-feature

2. 进行代码修改（遵循代码规范）

3. 运行测试确保通过
   pytest tests/ -v

4. 提交更改（遵循提交信息格式）
   git commit -m "feat: add flash attention support"

5. 推送分支
   git push origin feature/your-feature

6. 创建 Pull Request
```

---

## 代码规范

### Python 代码风格

本项目遵循 PEP 8 规范，并使用以下工具：

```bash
# 格式化代码
black triton_ops/ tests/ examples/

# 检查并修复 lint 问题
ruff check --fix triton_ops/ tests/ examples/

# 类型检查
mypy triton_ops/
```

### 文档字符串

使用 Google 风格的 docstring：

```python
def fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply fused RMSNorm + RoPE transformation.

    Combines RMSNorm and Rotary Position Embedding into a single
    kernel launch, reducing memory bandwidth requirements.

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim].
        weight: RMSNorm weight of shape [hidden_dim].
        cos: Cosine position embeddings of shape [seq_len, head_dim].
        sin: Sine position embeddings of shape [seq_len, head_dim].
        eps: Small constant for numerical stability. Defaults to 1e-6.

    Returns:
        Output tensor of shape [batch, seq_len, hidden_dim].

    Raises:
        DeviceError: If CUDA is not available.
        ShapeMismatchError: If tensor shapes are incompatible.

    Example:
        >>> x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
        >>> weight = torch.ones(4096, device='cuda', dtype=torch.float16)
        >>> cos = torch.randn(128, 64, device='cuda', dtype=torch.float16)
        >>> sin = torch.randn(128, 64, device='cuda', dtype=torch.float16)
        >>> output = fused_rmsnorm_rope(x, weight, cos, sin)
    """
```

### 类型注解

所有公开 API 必须包含类型注解：

```python
from typing import Literal, Optional

def fused_gated_mlp(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    activation: Literal["silu", "gelu"] = "silu",
) -> torch.Tensor:
    ...
```

### Triton Kernel 规范

新增 Triton kernel 时：

1. **添加参考实现** — 提供纯 PyTorch 的参考实现用于正确性验证
2. **添加正确性测试** — 使用 Hypothesis 进行基于属性的测试
3. **添加基准测试** — 放入 `tests/benchmarks/`
4. **文档说明** — 解释算法和优化策略

```python
@triton.jit
def my_kernel(
    input_ptr,
    output_ptr,
    # ... 参数
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel description.

    Algorithm:
        1. Step one
        2. Step two

    Optimization:
        - Uses shared memory for ...
        - Vectorized loads for ...
    """
```

---

## 测试要求

### 测试覆盖率

| 模块 | 最低覆盖率 |
|------|-----------|
| kernels/ | 90% |
| autotuner/ | 80% |
| benchmark/ | 70% |

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行带覆盖率
pytest tests/ -v --cov=triton_ops --cov-report=html

# 运行特定测试文件
pytest tests/test_rmsnorm_rope.py -v

# 运行基准测试（需要 CUDA）
python -m tests.benchmarks.bench_rmsnorm_rope

# Hypothesis 测试（CI 模式）
pytest tests/ --hypothesis-profile=ci
```

### 测试编写规范

```python
import pytest
import torch
from hypothesis import given, strategies as st

from triton_ops import fused_rmsnorm_rope


class TestFusedRMSNormRoPE:
    """Tests for fused RMSNorm + RoPE kernel."""

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("seq_len", [128, 512, 1024])
    def test_correctness(self, batch_size: int, seq_len: int):
        """Test output correctness against reference implementation."""
        # Setup
        x = torch.randn(batch_size, seq_len, 4096, device='cuda', dtype=torch.float16)
        # ... test implementation

    @given(
        batch_size=st.integers(min_value=1, max_value=16),
        seq_len=st.integers(min_value=1, max_value=1024),
    )
    def test_property_based(self, batch_size: int, seq_len: int):
        """Property-based test using Hypothesis."""
        # ...
```

---

## 提交信息格式

本项目采用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

### 格式

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### 类型

| 类型 | 描述 | 示例 |
|------|------|------|
| `feat` | 新功能 / 新算子 | `feat: add flash attention kernel` |
| `fix` | Bug 修复 | `fix: correct RMSNorm overflow handling` |
| `perf` | 性能优化 | `perf: optimize FP8 GEMM block sizes` |
| `refactor` | 代码重构 | `refactor: simplify autotuner interface` |
| `docs` | 文档更新 | `docs: add API reference for fp8_gemm` |
| `test` | 测试相关 | `test: add edge case tests for RoPE` |
| `chore` | 杂项（构建、依赖等） | `chore: update CI workflow` |
| `style` | 代码格式 | `style: format with black` |

### Scope（可选）

| Scope | 描述 |
|-------|------|
| `kernels` | 内核实现 |
| `autotuner` | 自动调优 |
| `benchmark` | 基准测试 |
| `fp8` | FP8 相关 |
| `rmsnorm` | RMSNorm 相关 |
| `rope` | RoPE 相关 |
| `mlp` | MLP 相关 |

### 示例

```bash
# 新功能
feat(kernels): add fused attention kernel

Add fused attention kernel that combines QKV projection and
scaled dot-product attention into a single kernel.

Closes #42

# Bug 修复
fix(fp8): handle overflow in quantize_fp8

The previous implementation did not properly handle tensors with
values exceeding FP8 max (448.0). Now uses dynamic scaling with
automatic retry.

Fixes #89

# 性能优化
perf(kernels): use adaptive block sizes in fp8_gemm

Replace hardcoded block sizes with problem-size-aware heuristics
for better performance across different matrix dimensions.

Benchmarks show 15% improvement for small matrices.
```

---

## Pull Request 流程

### PR 检查清单

在提交 PR 前，请确保：

- [ ] 代码通过所有测试 (`pytest tests/ -v`)
- [ ] 代码通过 lint 检查 (`ruff check triton_ops/ tests/`)
- [ ] 代码已格式化 (`black triton_ops/ tests/`)
- [ ] 新功能有对应的测试
- [ ] 新功能有对应的文档
- [ ] 提交信息符合 Conventional Commits 规范
- [ ] PR 描述清晰说明了改动内容

### PR 模板

```markdown
## 描述

简要描述此 PR 的内容和目的。

## 改动类型

- [ ] 新功能 (feat)
- [ ] Bug 修复 (fix)
- [ ] 性能优化 (perf)
- [ ] 文档更新 (docs)
- [ ] 代码重构 (refactor)
- [ ] 测试 (test)

## 改动内容

- 添加了 xxx 功能
- 修复了 xxx 问题
- 更新了 xxx 文档

## 测试

描述如何测试这些改动：

```bash
pytest tests/test_xxx.py -v
```

## 相关 Issue

Closes #xxx
```

### 代码审查

所有 PR 都需要至少一位维护者的审查批准。审查重点：

1. **正确性** — 代码逻辑是否正确
2. **性能** — 是否有性能回归
3. **可维护性** — 代码是否清晰易读
4. **测试覆盖** — 测试是否充分
5. **文档** — 文档是否完整

---

## 文档贡献

### 文档结构

```
docs/
├── api/              # API 参考文档
├── guides/           # 使用指南
├── internals/        # 内部实现说明
└── examples/         # 详细示例

README.md             # 项目概述
README.zh-CN.md       # 中文版 README
CONTRIBUTING.md       # 本文档
CHANGELOG.md          # 变更日志
```

### 文档风格

- 使用简洁清晰的语言
- 提供可运行的代码示例
- 包含必要的数学公式（使用 LaTeX）
- 添加图表说明复杂概念

### 构建文档

```bash
# 本地预览 GitHub Pages
bundle exec jekyll serve

# 或使用 Python
pip install mkdocs
mkdocs serve
```

---

## 获取帮助

- **GitHub Issues**: [提交问题](https://github.com/LessUp/triton-fused-ops/issues)
- **Discussions**: [参与讨论](https://github.com/LessUp/triton-fused-ops/discussions)
- **Email**: triton-fused-ops@example.com

---

再次感谢你的贡献！ 🎉
