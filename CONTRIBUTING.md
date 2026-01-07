# Contributing to Triton Fused Operators

Thank you for your interest in contributing to Triton Fused Operators! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Message Format](#commit-message-format)
- [Reporting Issues](#reporting-issues)

## Development Setup

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (for running tests)
- Git

### Setting Up Your Development Environment

1. **Fork the repository**

   Click the "Fork" button on GitHub to create your own copy of the repository.

2. **Clone your fork**

   ```bash
   git clone https://github.com/YOUR_USERNAME/triton-fused-ops.git
   cd triton-fused-ops
   ```

3. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**

   ```bash
   # Install in development mode with all dependencies
   pip install -e ".[dev]"
   ```

5. **Verify installation**

   ```bash
   # Run tests to verify everything is working
   pytest tests/ -v
   ```

## Code Style

We use several tools to maintain consistent code quality:

### Formatting with Black

All Python code must be formatted with [Black](https://black.readthedocs.io/):

```bash
# Format all files
black .

# Check formatting without making changes
black --check .
```

### Linting with Ruff

We use [Ruff](https://docs.astral.sh/ruff/) for fast Python linting:

```bash
# Run linting
ruff check .

# Auto-fix issues where possible
ruff check --fix .
```

### Type Checking with Mypy

All code must pass [Mypy](https://mypy.readthedocs.io/) type checking:

```bash
mypy triton_ops/
```

### Docstrings

We follow [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings):

```python
def example_function(param1: int, param2: str) -> bool:
    """Brief description of the function.

    Longer description if needed, explaining the function's behavior
    in more detail.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When param1 is negative.

    Example:
        >>> example_function(1, "test")
        True
    """
    ...
```

### Type Hints

All public functions and methods must have complete type annotations:

```python
from typing import Optional, List

def process_data(
    inputs: List[torch.Tensor],
    scale: Optional[float] = None,
) -> torch.Tensor:
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_rmsnorm_rope.py -v

# Run with coverage
pytest tests/ -v --cov=triton_ops --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test function names
- Include both unit tests and property-based tests where appropriate

```python
import pytest
import torch
from triton_ops import rmsnorm_rope_fused

def test_rmsnorm_rope_basic():
    """Test basic RMSNorm + RoPE functionality."""
    # Setup
    x = torch.randn(2, 4, 8, 64, device="cuda", dtype=torch.float16)
    cos = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    sin = torch.randn(4, 64, device="cuda", dtype=torch.float16)
    weight = torch.ones(64, device="cuda", dtype=torch.float16)
    
    # Execute
    result = rmsnorm_rope_fused(x, weight, cos, sin)
    
    # Verify
    assert result.shape == x.shape
    assert result.dtype == x.dtype
```

## Pull Request Process

### Before Submitting

1. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**

   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation if needed

3. **Run quality checks**

   ```bash
   # Format code
   black .
   
   # Run linting
   ruff check .
   
   # Run type checking
   mypy triton_ops/
   
   # Run tests
   pytest tests/ -v
   ```

4. **Commit your changes**

   Follow the [commit message format](#commit-message-format).

### Submitting the PR

1. Push your branch to your fork
2. Open a Pull Request against the `main` branch
3. Fill out the PR template completely
4. Wait for CI checks to pass
5. Address any review feedback

### PR Review Criteria

- All CI checks must pass
- Code follows the style guidelines
- Tests are included for new functionality
- Documentation is updated if needed
- Changes are focused and atomic

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(kernels): add FP8 GEMM kernel with per-tensor scaling

fix(rmsnorm): correct epsilon handling for edge cases

docs(readme): add performance comparison table

test(gated-mlp): add property tests for activation functions
```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- Python version
- PyTorch version
- Triton version
- GPU model and CUDA version
- Minimal reproducible example
- Expected vs actual behavior

### Feature Requests

For feature requests, please describe:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## Questions?

If you have questions, feel free to:

- Open a [Discussion](https://github.com/username/triton-fused-ops/discussions)
- Check existing [Issues](https://github.com/username/triton-fused-ops/issues)

Thank you for contributing! 🎉
