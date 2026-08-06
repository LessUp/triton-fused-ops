# Triton Fused Ops

面向 AI Infra 学习的精简 Triton 算子仓库。只保留三条可以用独立参考实现验证的 Transformer 推理路径：

- `fused_rmsnorm_rope`：融合 RMSNorm 与 RoPE
- `fused_gated_mlp`：标准 SwiGLU/GeGLU，公式为 `activation(gate_proj(x)) * up_proj(x)`
- `flash_attention`：带在线 softmax 的 FlashAttention 前向，支持 causal mask

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

## 项目边界

这个仓库练习 Triton kernel 与验证方法，不承担以下职责：

- CUDA C++ 的系统学习路径：[`cuda-kernel-academy`](https://github.com/AICL-Lab/cuda-kernel-academy)
- FlashAttention 前后向的 CUDA C++ 深挖：[`cuflash-attn`](https://github.com/AICL-Lab/cuflash-attn)
- 完整模型加载与 token 生成：[`tiny-llm`](https://github.com/AICL-Lab/tiny-llm)
- Paged KV 与 continuous batching 控制面：[`paged-infer`](https://github.com/AICL-Lab/paged-infer)

新 kernel 只有在具备独立参考实现、边界测试和真实 GPU 验证计划时才进入主分支。未在当前硬件上测量的性能数字不会写入 README。

## License

MIT，详见 [LICENSE](LICENSE)。
