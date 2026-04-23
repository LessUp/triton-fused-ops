# Git hooks

This repository provides deterministic local hooks for touched files.

## Enable hooks

```bash
git config core.hooksPath .githooks
```

## Current hooks

- `pre-commit`: runs `ruff format` and `ruff check --fix` on staged `*.py` files, then re-stages them.
