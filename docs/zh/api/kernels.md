---
title: 核心算子
description: "融合 kernel 入口函数与模块封装参考"
---

# 核心算子

本页记录 `triton_ops` 当前导出的主要计算接口。

## `fused_rmsnorm_rope`

```python
fused_rmsnorm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
    num_heads: int | None = None,
) -> torch.Tensor
```

用途：

- 在一次 kernel 启动中完成 RMSNorm 与 RoPE。
- 避免把归一化后的中间结果单独写回 HBM。

输入契约：

- `x` 必须是 contiguous 的 CUDA 张量，形状为 `[batch, seq_len, hidden_dim]`。
- `weight` 必须是 contiguous 的 CUDA 张量，形状为 `[hidden_dim]`。
- `cos` 与 `sin` 必须是形状一致的 contiguous CUDA 张量。
- 当前支持的 RoPE cache 形状：
  - `[seq_len, head_dim]`
  - `[1, seq_len, 1, head_dim]`
- `head_dim` 必须是偶数。
- 如果不传 `num_heads`，会按 `hidden_dim / head_dim` 自动推断。

输出：

- 形状与 `x` 相同。
- dtype 与 `x` 相同。

常见错误：

- `DeviceError`：输入不在 CUDA 上。
- `ShapeMismatchError`：形状不匹配。
- `UnsupportedDtypeError`：dtype 不在支持范围内。

## `FusedRMSNormRoPE`

```python
FusedRMSNormRoPE(hidden_dim: int, head_dim: int, eps: float = 1e-6)
```

该模块内部持有 RMSNorm 的权重参数，但前向仍然要求显式传入 `cos` 与 `sin`：

```python
module = FusedRMSNormRoPE(4096, 64).cuda()
out = module(x, cos, sin)
```

集成提醒：

- 它不是普通 LayerNorm/RMSNorm 的直接替代品，因为前向契约包含 RoPE 输入。

## `fused_gated_mlp`

```python
fused_gated_mlp(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    activation: Literal["silu", "gelu"] = "silu",
) -> torch.Tensor
```

当前 Triton kernel 与参考实现都遵循：

```text
output = activation(gate_proj(x)) * up_proj(x)
```

输入契约：

- `x`：contiguous CUDA 张量，形状 `[batch, seq_len, hidden_dim]`
- `gate_weight`：contiguous CUDA 张量，形状 `[intermediate_dim, hidden_dim]`
- `up_weight`：形状必须与 `gate_weight` 相同
- `activation`：只能是 `"silu"` 或 `"gelu"`

输出：

- 形状 `[batch, seq_len, intermediate_dim]`
- dtype 与 `x` 相同

重要边界：

- 该 kernel 只覆盖 gated expansion 这一步。
- 完整 Transformer MLP 仍然需要在外部补上下投影与 residual 逻辑。

## `FusedGatedMLP`

```python
FusedGatedMLP(
    hidden_dim: int,
    intermediate_dim: int,
    activation: Literal["silu", "gelu"] = "silu",
)
```

这个模块内部持有 `gate_weight` 与 `up_weight`，前向时会调用 `fused_gated_mlp`。

## `fp8_gemm`

```python
fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor | None = None,
    b_scale: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor
```

行为说明：

- 如果 `a` 或 `b` 还是浮点张量，函数会先内部调用 `quantize_fp8`。
- 如果输入已经是仓库定义的 FP8 存储格式，则必须同时提供对应 scale。
- 当前维护的运行路径使用基于 `uint8` 的 FP8 兼容表示。

输入契约：

- `a` 与 `b` 必须是 contiguous 的 CUDA 张量。
- 矩阵形状必须分别为 `[M, K]` 与 `[K, N]`。
- 预量化输入需要在同一设备上提供标量 scale 张量。

输出：

- 形状 `[M, N]`
- 实际使用时请优先视为 `torch.float16` 或 `torch.bfloat16` 输出路径

实践提醒：

- 校验层允许 `torch.float32` 作为 `output_dtype`，但 Triton 实现的维护重点仍是半精度输出路径。实践中应优先使用 `float16` / `bfloat16`。

## `FP8Linear`

```python
FP8Linear(in_features: int, out_features: int, bias: bool = False)
```

行为：

- 内部保留一个可训练的浮点 `weight` 参数。
- 第一次前向时，量化并缓存：
  - `weight_fp8`
  - `weight_scale`
  - `weight_fp8_t`（转置后且 contiguous）
- 前向调用时使用 `fp8_gemm` 完成计算。

重要集成提醒：

- 缓存后的 FP8 权重不会在权重更新后自动刷新。
- 因此 `FP8Linear` 更适合推理场景，或者权重稳定的阶段。
