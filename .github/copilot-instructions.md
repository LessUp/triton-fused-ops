# Copilot instructions for `triton-fused-ops`

## Repository priorities

1. Keep kernel behavior correct and claims evidence-backed.
2. Keep docs/workflows concise, project-specific, and maintainable.
3. Prefer high-signal automation over broad noisy pipelines.

## OpenSpec-first contract

- For non-trivial work, always start from an active OpenSpec change.
- Complete proposal/design/specs/tasks before implementation.
- Implement in task order and update task checkboxes immediately.
- If design/scope shifts, update OpenSpec artifacts first.

## Expected validation baseline

```bash
ruff format --check .
ruff check .
mypy triton_ops/
pytest tests/ -v -k "not cuda and not gpu" --ignore=tests/benchmarks/
python3 -m build
```

## Collaboration and review

- Use `/review` at integration boundaries and before merge.
- Use subagents for bounded parallel scopes; avoid duplicate investigation.
- Prefer long coherent sessions over fragmented micro-runs.
- Avoid high-cost modes unless lower-cost/default execution cannot complete the task.

## GitHub-centric operations

- Use `gh` for repository metadata, issues, PRs, and workflow inspection.
- Keep GitHub About metadata (description/homepage/topics) aligned with README and Pages positioning.
- When using `/remote` or `/research`, capture only decisions that become actionable in this repository.
