---
layout: default
title: "v0.2.0 — Major Refactoring (2026-03-09)"
description: "SwiGLU 正确性修复、FP8Linear 权重转置缓存、RMSNorm batch_idx 修复"
---

[← 返回更新日志](./)

# Major Refactoring - v0.2.0

**发布日期:** 2026-03-09

---

## 🐛 关键 Bug 修复

### Gated MLP: gate/up 投影激活顺序错误

**问题:** 标准 SwiGLU 公式为 `output = activation(gate_proj(x)) * up_proj(x)`，但内核和参考实现错误地将激活函数应用于 `up_acc` 而非 `gate_acc`。

**影响:** 所有使用 SwiGLU 的模型（LLaMA、Mistral 等）输出错误。

**修复:**
- `triton_ops/kernels/gated_mlp.py` — 激活函数现在正确应用于 gate 投影
- `gated_mlp_reference()` — 参考实现同步修复

### RMSNorm 内核: batch_idx 计算错误

**问题:** `rmsnorm_kernel` 计算 `batch_idx = row_idx // cdiv(hidden_dim, BLOCK_SIZE)` 数学上错误 — `hidden_dim / BLOCK_SIZE` 与批索引无关。

**影响:** 该变量未被使用，不影响输出正确性，但表明对程序网格的理解有误。

**修复:** 移除错误的计算；`row_idx` 从 `program_id(0)` 已经正确索引扁平化的 batch*seq 网格。

---

## ⚡ 性能优化

### FP8Linear: 权重转置缓存

**问题:** `FP8Linear.forward()` 每次前向都调用 `self.weight_fp8.t().contiguous()`，对大权重矩阵是昂贵的分配+拷贝操作。

**修复:** 在 `quantize_weights()` 时预计算并缓存转置后的权重为 `weight_fp8_t`。

**收益:** 消除每次前向传递的一次 GPU 分配。

```python
# 之前: 每次前向都转置
def forward(self, x):
    output = fp8_gemm(x_fp8, self.weight_fp8.t().contiguous(), ...)  # 昂贵!

# 之后: 使用缓存的转置权重
def quantize_weights(self):
    self.weight_fp8_t = weight_fp8.t().contiguous()  # 只转置一次

def forward(self, x):
    output = fp8_gemm(x_fp8, self.weight_fp8_t, ...)  # 使用缓存
```

---

## 🧹 代码质量

### api.py: 合并重复导入

每个内核模块之前被两个独立的 `from...import` 块导入，已合并为单一导入语句。

---

## 📁 变更文件

| 文件 | 变更类型 |
|------|----------|
| `triton_ops/kernels/gated_mlp.py` | Bug 修复 — 激活函数应用顺序 |
| `triton_ops/kernels/fp8_gemm.py` | 性能优化 — 缓存转置权重 |
| `triton_ops/kernels/rmsnorm_rope.py` | Bug 修复 — 移除错误计算 |
| `triton_ops/api.py` | 重构 — 合并导入 |
| `triton_ops/__init__.py` | 版本更新 |
| `pyproject.toml` | 版本更新 |

---

## 版本信息

- **前一版本:** v0.1.0
- **当前版本:** v0.2.0
- **比较链接:** [v0.1.0...v0.2.0](https://github.com/LessUp/triton-fused-ops/compare/v0.1.0...v0.2.0)
