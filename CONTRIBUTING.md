# Contributing

Thanks for contributing to `triton-fused-ops`.

This repository is maintained with a **spec-driven, quality-first** workflow.

## 1. Code of conduct

Please follow the [Code of Conduct](.github/CODE_OF_CONDUCT.md).

## 2. Local setup

```bash
git clone https://github.com/LessUp/triton-fused-ops.git
cd triton-fused-ops
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
git config core.hooksPath .githooks
```

## 3. Required workflow (OpenSpec-first)

For non-trivial work (cross-file changes, behavior changes, workflow changes):

1. Create/select an OpenSpec change in `openspec/changes/<change-name>/`.
2. Complete `proposal.md`, `design.md`, `tasks.md`, and required specs.
3. Implement tasks in order and mark checkboxes immediately.
4. Open/update a PR and run review.
5. Archive the OpenSpec change after merge.

Common command shortcuts in this repo:

```bash
/opsx:explore
/opsx:propose
/opsx:apply
/opsx:archive
```

## 4. Validation baseline

Run this baseline before PR updates:

```bash
ruff format --check .
ruff check .
mypy triton_ops/
pytest tests/ -v -k "not cuda and not gpu" --ignore=tests/benchmarks/
python3 -m build
```

If your changes affect CUDA kernels, also run the relevant GPU correctness/performance checks on a CUDA-capable machine.

## 5. Style and scope rules

- Keep diffs focused and easy to review.
- Keep public README/docs claims evidence-backed.
- Do not introduce planning systems outside OpenSpec.
- Keep CI/workflow edits high-signal and minimal.
- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, etc.).
- Keep staged Python files hook-clean (`ruff format` + `ruff check --fix`).

## 6. Pull request expectations

- Link the OpenSpec change in the PR description.
- Summarize what changed and why.
- Call out risks/trade-offs explicitly.
- Keep branches short-lived and merge quickly after review.

## 7. Assistant workflow notes

- Use `/review` for deeper analysis at integration boundaries.
- Prefer default/lower-cost models first; use high-cost modes only when needed.
- Keep MCP-style integrations opt-in and justified by concrete needs.
