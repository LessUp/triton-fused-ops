---
name: verify
description: Run full CI verification (lint, typecheck, test with coverage). Use before committing or to validate changes.
---

Run the complete CI pipeline for this Triton ops project:

```bash
# Lint
ruff check triton_ops/ tests/ examples/

# Format check
ruff format --check triton_ops/ tests/ examples/

# Type check
mypy triton_ops/ --ignore-missing-imports

# Tests with coverage
pytest tests/ -v --cov=triton_ops --cov-report=term-missing
```

If running on a CPU-only machine, note that GPU/CUDA tests will be skipped.
