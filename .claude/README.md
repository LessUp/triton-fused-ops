# Claude workspace notes

This directory contains Claude command/skill assets used in this repository.

## Workflow alignment

- Canonical repo guidance is in root `AGENTS.md` and `CLAUDE.md`.
- Non-trivial work follows OpenSpec-first flow:
  1. `/opsx:explore` (clarify)
  2. `/opsx:propose` (create change artifacts)
  3. `/opsx:apply` (implement tasks)
  4. `/review` (integration review)
  5. `/opsx:archive` (after merge)

## Tooling posture

- Prefer default/lower-cost models for routine implementation.
- Use expensive modes only when necessary for quality or blocker resolution.
- Keep MCP integrations optional and justified by concrete need.

## Local settings

- `settings.local.json` is local-machine behavior and may vary by environment.
- Local overrides should not weaken repository quality gates.
