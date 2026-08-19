# 变更日志

## Unreleased

- 面向用户的 GitHub 链接与 `pyproject.toml` 仓库地址统一为 `github.com/aicl-lab/...`

## 2.0.0 - 2026-08-06

- 合入经过 PyTorch SDPA 对照验证的 Triton FlashAttention 前向实现。
- 统一 Gated MLP 为标准 SwiGLU/GeGLU 契约：`activation(gate) * up`。
- 删除错误标注为 FP8 E4M3 的线性 `uint8` 量化、GEMM、测试、示例和文档。
- 删除双语文档站、OpenSpec、AI 工具配置、容器与 CI 样板，收敛为核心代码、参考实现、测试和中文 README。
- 更新仓库地址到 AICL-Lab，并移除未经当前硬件复测的性能承诺。
