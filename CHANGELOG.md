# 变更日志

## Unreleased

## 2.0.1 - 2026-08-23

- 面向用户的 GitHub 链接与 `pyproject.toml` 仓库地址统一为 `github.com/open-infra-ai/...`。
- 新增 CPU GitHub Actions 门禁与手动自托管 GPU 验证工作流，覆盖 Ruff、mypy、
  CPU 测试、Hypothesis CI profile 和包构建。
- 将占位团队与 `example.com` 邮箱替换为公开维护者身份 `LessUp`，补充 Python
  3.12 分类和 `build` 开发依赖。
- 删除没有任何测试使用的 `--gpu` / `--slow` pytest 开关；GPU 用例继续通过
  `torch.cuda.is_available()` 明确跳过。
- 补齐历史 `v2.0.0` tag，并发布 `v2.0.1` 维护版本，使源码包、CHANGELOG 与
  GitHub Release 恢复一致；README 记录 2026-08-23 的 CPU-only
  57 passed / 66 skipped 与 GPU 123/123 结果。
- README 首屏补充“Transformer 推理融合算子 + `torch.library` 集成”定位，
  保留已建立引用的仓库名，只优化展示文案。

## 2.0.0 - 2026-08-06

- 合入经过 PyTorch SDPA 对照验证的 Triton FlashAttention 前向实现。
- 统一 Gated MLP 为标准 SwiGLU/GeGLU 契约：`activation(gate) * up`。
- 删除错误标注为 FP8 E4M3 的线性 `uint8` 量化、GEMM、测试、示例和文档。
- 删除双语文档站、OpenSpec、AI 工具配置、容器与 CI 样板，收敛为核心代码、参考实现、测试和中文 README。
- 更新仓库地址到 AICL-Lab，并移除未经当前硬件复测的性能承诺。
