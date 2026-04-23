# CLAUDE.md

Repository-specific operating guidance for Claude-based workflows.

## Working mode

- Treat this repo as a **stabilization/finish-line** project.
- Make focused, high-signal edits; avoid broad speculative refactors.
- Keep docs, workflows, and metadata aligned with actual project behavior.

## OpenSpec contract

- Start non-trivial work from an active OpenSpec change.
- Do not implement before proposal/design/tasks are ready.
- While implementing, update task checkboxes as soon as a task is complete.
- If implementation reveals scope/design drift, update artifacts before continuing.

## Review and execution

- Use `/review` before merge and after major task groups.
- Prefer one coherent branch per change and merge quickly.
- Use subagents for bounded scopes; avoid redundant parallel work.
- Use higher-cost/fleet-style modes only when cheaper modes cannot complete the task.

## Local validation baseline

```bash
ruff format --check .
ruff check .
mypy triton_ops/
pytest tests/ -v -k "not cuda and not gpu" --ignore=tests/benchmarks/
python3 -m build
```

## Local overrides

- Personal local preferences can live in `CLAUDE.local.md`.
- `CLAUDE.local.md` must not contradict repository-level quality and workflow gates.
