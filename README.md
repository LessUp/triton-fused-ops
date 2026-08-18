# Triton Fused Ops

面向 AI Infra 学习的精简 Triton 算子仓库。只保留三条可以用独立参考实现验证的 Transformer 推理路径：

- `fused_rmsnorm_rope`：融合 RMSNorm 与 RoPE
- `fused_gated_mlp`：标准 SwiGLU/GeGLU，公式为 `activation(gate_proj(x)) * up_proj(x)`
- `flash_attention`：带在线 softmax 的 FlashAttention 前向，支持 causal mask

> ℹ️ **定位**：Triton FlashAttention 是 [cuflash-attn](https://github.com/AICL-Lab/cuflash-attn)
> 的独立参考实现，用于验证 CUDA C++ 版本的正确性。完整 FlashAttention 前后向 +
> 优化叙事见 cuflash-attn（本仓库只保留前向参考实现）。

仓库同时保留 NumPy/PyTorch 参考实现、输入契约、差分测试、benchmark 与 autotuner 基础设施。旧版所谓“FP8 E4M3”实际是存储在 `uint8` 中的均匀线性量化，并不编码 E4M3 指数和尾数，因此已整条删除，避免把 INT8 路径误当作 FP8 教材。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

运行 Triton kernel 需要 NVIDIA GPU、CUDA 版 PyTorch 和 Triton。CPU 环境仍可运行参考实现、输入契约、格式和打包检查。

## 快速示例

```python
import torch
from triton_ops import flash_attention, fused_gated_mlp, fused_rmsnorm_rope

x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.float16)
gate_weight = torch.randn(11008, 4096, device="cuda", dtype=torch.float16)
up_weight = torch.randn_like(gate_weight)

# 标准 SwiGLU: silu(gate_proj(x)) * up_proj(x)
mlp_output = fused_gated_mlp(x, gate_weight, up_weight, activation="silu")

q = torch.randn(2, 32, 128, 128, device="cuda", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
attention_output = flash_attention(q, k, v, causal=True)
```

RMSNorm + RoPE 的完整调用示例见 [`examples/rmsnorm_rope_example.py`](examples/rmsnorm_rope_example.py)，Gated MLP 示例见 [`examples/gated_mlp_example.py`](examples/gated_mlp_example.py)。

## torch.library 自定义算子

`import triton_ops` 会把三个公开 kernel 注册进 `torch.ops.triton_ops.*` 命名空间
（与 vLLM/SGLang 等推理框架用 `torch.library` 注册自定义 op 的方式一致，便于接入
`torch.compile` / `torch.export` 图）：

```python
import torch
import triton_ops  # 触发注册

# triton_ops::sgemm(Tensor a, Tensor b) -> Tensor
c = torch.ops.triton_ops.sgemm(a, b)

# triton_ops::fused_rmsnorm_rope(Tensor x, Tensor weight, Tensor cos, Tensor sin, float eps) -> Tensor
out = torch.ops.triton_ops.fused_rmsnorm_rope(x, weight, cos, sin)

# triton_ops::fused_gated_mlp(Tensor x, Tensor gate_weight, Tensor up_weight, str activation) -> Tensor
out = torch.ops.triton_ops.fused_gated_mlp(x, gate_weight, up_weight, activation="silu")
```

注册策略（`triton_ops/ops.py`）：

- 优先 `torch.library.triton_op`（torch 2.13+）：实现由 Triton kernel 构成，
  对 `torch.compile`/`torch.export` 可见，可被进一步优化；
- 不可用时 fallback 到 `torch.library.custom_op + register_fake`：把 op 当 opaque，
  提供 eager 执行与 shape 推断，兼容旧版 PyTorch。

所有 op 只接受 CUDA 张量，CPU 输入直接抛 `NotImplementedError`；op 内部只调用
`triton_ops.kernels.*` 的公开函数，不复制 kernel 逻辑。

### 与 vLLM / SGLang custom op 的对应关系

推理框架用同一套 `torch.library` 机制暴露自定义 kernel：

- vLLM 的 `_custom_ops.py` 用 `torch.library.custom_op` 注册 `vllm::*` 命名空间
  （如 `vllm::attention.forward`），并用 `torch.library.register_fake` 提供 meta 实现；
- SGLang 通过 `torch.library` 暴露 `sglang::*` 算子，同样以 `register_fake` 支持
  torch.compile / 图捕获；
- 本仓库用同一模式：`triton_ops::sgemm` / `triton_ops::fused_rmsnorm_rope` /
  `triton_ops::fused_gated_mlp`。差别仅在实现层：vLLM/SGLang 的生产 kernel 走
  FlashInfer/CUDA Graph 等，这里用 Triton kernel 做最小可验证实现。

## 验证

```bash
ruff format --check .
ruff check .
mypy triton_ops --ignore-missing-imports
pytest -q
python -m build
```

测试分为两层：

- CPU 可运行：NumPy/PyTorch 参考模型、标准 SwiGLU 契约、FlashAttention/SDPA 对照、输入失败路径、benchmark/autotuner 工具。
- 必须有 CUDA：Triton kernel 与参考实现的数值差分测试；无 GPU 时明确 skip，不报告为已通过。

## 性能基准（真实 GPU 实测）

> 环境：**RTX 3060 Laptop（`sm_86`，6144 MiB）**，驱动 591.44，CUDA 12.1
> （torch cu121），PyTorch **2.5.1**，Triton **3.1.0**，numpy 2.4.6，commit `ebf6c32+`。
> 计时：CUDA 同步墙钟，预热后取中位数。数值为稳态延迟，与参考实现差分验证
> 通过（`rtol=1e-2, atol=1e-2`，与测试套件一致）。

| 算子 | 配置 (M/batch, seq, hidden, inter) | 延迟 (ms) | 说明 |
|------|-------------------------------------|-----------|------|
| `fused_gated_mlp` (silu) | (1, 128, 4096, 11264) | **3.45** | 3 个 GEMM + SwiGLU |
| `fused_gated_mlp` (gelu) | (1, 128, 4096, 11264) | **3.50** | 同上，gelu 用 tanh 近似 |
| `fused_rmsnorm_rope` | (1, 128, 4096) | **0.104** | elementwise，带宽受限 |
| `fused_rmsnorm_rope` | (1, 512, 4096) | **0.237** | |
| `fused_rmsnorm_rope` | (4, 128, 4096) | **0.215** | |
| `fused_rmsnorm_rope` | (4, 512, 4096) | **0.682** | |

复现（仓库内）：

```bash
# 需要 CUDA GPU + torch/triton
python -m tests.benchmarks.bench_gated_mlp
python -m tests.benchmarks.bench_rmsnorm_rope
# 或用 BenchmarkSuite 定制配置：
#   from triton_ops.benchmark import BenchmarkSuite
#   BenchmarkSuite(warmup_runs=3, benchmark_runs=20).benchmark_gated_mlp(...)
```

> 说明：`fused_gated_mlp` 的 FLOPs 为 3 个 GEMM（2×[M,K=4096,N=11264] +
> 1×[M,K=11264,N=4096]）≈ 3.5e10，3.45ms 对应约 10 TFLOPS（RTX 3060 Laptop
> FP16 理论峰值约 46 TFLOPS）。gelu 的 tanh 近似在个别元素上与 numpy 参考
> 有 <1% 偏差（2/1.4M 元素超出 `rtol=1e-2, atol=1e-2`），数值正确性由
> `tests/test_gated_mlp.py` 的差分测试覆盖。

## 项目边界

这个仓库练习 Triton kernel 与验证方法，不承担以下职责：

- CUDA C++ 的系统学习路径：[`cuda-foundations`](https://github.com/AICL-Lab/cuda-foundations)
- FlashAttention 前后向的 CUDA C++ 深挖：[`cuflash-attn`](https://github.com/AICL-Lab/cuflash-attn)
- 完整模型加载与 token 生成：[`tiny-llm`](https://github.com/AICL-Lab/tiny-llm)
- Paged KV 与 continuous batching 控制面：[`paged-infer`](https://github.com/AICL-Lab/paged-infer)

新 kernel 只有在具备独立参考实现、边界测试和真实 GPU 验证计划时才进入主分支。未在当前硬件上测量的性能数字不会写入 README。

## License

MIT，详见 [LICENSE](LICENSE)。
