# OpenSpec workflow for this repository

This repository uses OpenSpec as the single source of truth for non-trivial work.

## Required flow

1. `openspec new change <name>` (or continue an existing active change)
2. Complete artifacts (`proposal`, `design`, `specs`, `tasks`)
3. Implement tasks in order and keep `tasks.md` checkboxes current
4. Run review and validation commands
5. Merge, then archive the change

## Policy

- Non-trivial implementation MUST start from an active OpenSpec change.
- Do not use parallel planning/spec systems in the mainline repository.
- If execution reveals design drift, update OpenSpec artifacts before continuing.
