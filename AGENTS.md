# AGENTS.md

This repository uses an **OpenSpec-first** execution model.

## Project intent

- Keep `triton-fused-ops` stable, credible, and easy to maintain.
- Prioritize correctness, documentation quality, and workflow signal over feature expansion.
- Keep user-facing claims evidence-backed.

## Mandatory workflow (non-trivial work)

1. Create or select an OpenSpec change.
2. Ensure `proposal.md`, `design.md`, `tasks.md`, and required specs are complete.
3. Implement tasks in order, marking checkboxes immediately.
4. Run `/review` (or equivalent review step) before merge.
5. Merge quickly; avoid long-lived branch drift.

For OpenSpec actions in this repo:
- Propose: `/opsx:propose`
- Explore/clarify: `/opsx:explore`
- Implement: `/opsx:apply`
- Archive completed change: `/opsx:archive`

## Branch and merge policy

- One OpenSpec change per branch.
- Rebase or merge from `main` frequently.
- Avoid parallel local/cloud branches that sit unmerged.
- Prefer small, reviewable PRs that map cleanly to task groups.

## Quality baseline

Run these commands before PR/merge:

```bash
ruff format --check .
ruff check .
mypy triton_ops/
pytest tests/ -v -k "not cuda and not gpu" --ignore=tests/benchmarks/
python3 -m build
```

GPU validation can be run separately on CUDA-capable machines for kernel correctness/perf confidence.

## Tooling policy

- LSP/editor baseline: **Pylance/Pyright-compatible + Ruff + mypy**.
- MCP integrations are **opt-in** and must have a clear project-specific ROI.
- Prefer long coherent sessions over high-cost parallel experimentation.
- Use `/review` intentionally at integration boundaries.
