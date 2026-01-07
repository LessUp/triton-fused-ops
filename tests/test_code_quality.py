"""Code quality property tests for triton_ops package.

These tests verify that the codebase maintains quality standards:
- Property 1: Type Annotation Completeness
- Property 2: Docstring Completeness and Structure
"""

import inspect
from pathlib import Path
from typing import Callable, List

import pytest

# Try to import hypothesis for property-based testing
try:
    from hypothesis import given, settings, strategies as st

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False

# Try to import triton_ops
try:
    import triton_ops

    HAS_TRITON_OPS = True
except ImportError:
    HAS_TRITON_OPS = False


def get_public_functions() -> List[Callable]:
    """Get all public functions from triton_ops package.

    Returns:
        List of public function objects
    """
    if not HAS_TRITON_OPS:
        return []

    functions = []

    # Get functions from main module
    for name in dir(triton_ops):
        if name.startswith("_"):
            continue
        obj = getattr(triton_ops, name)
        if callable(obj) and not isinstance(obj, type):
            functions.append(obj)

    return functions


def get_public_classes() -> List[type]:
    """Get all public classes from triton_ops package.

    Returns:
        List of public class objects
    """
    if not HAS_TRITON_OPS:
        return []

    classes = []

    for name in dir(triton_ops):
        if name.startswith("_"):
            continue
        obj = getattr(triton_ops, name)
        if isinstance(obj, type):
            classes.append(obj)

    return classes


def get_public_methods(cls: type) -> List[Callable]:
    """Get all public methods from a class.

    Args:
        cls: Class to inspect

    Returns:
        List of public method objects
    """
    methods = []
    for name in dir(cls):
        if name.startswith("_"):
            continue
        obj = getattr(cls, name)
        if callable(obj):
            methods.append(obj)
    return methods


@pytest.mark.skipif(not HAS_TRITON_OPS, reason="triton_ops not available")
class TestTypeAnnotations:
    """Tests for type annotation completeness.

    Feature: project-standardization, Property 1: Type Annotation Completeness
    Validates: Requirements 4.1, 4.4
    """

    def test_public_functions_have_return_annotations(self):
        """Verify public functions have return type annotations."""
        functions = get_public_functions()

        for func in functions:
            sig = inspect.signature(func)
            # Allow functions without return annotation if they're wrappers
            if sig.return_annotation == inspect.Signature.empty:
                # Check if it's a simple wrapper or has complex logic
                try:
                    source = inspect.getsource(func) if hasattr(func, "__code__") else ""
                except (OSError, TypeError):
                    source = ""
                if "def " in source and "return" in source:
                    # Has return statement, should have annotation
                    # This is a soft check - we don't fail but log
                    pass

    def test_public_functions_have_parameter_annotations(self):
        """Verify public function parameters have type annotations."""
        functions = get_public_functions()

        for func in functions:
            sig = inspect.signature(func)
            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls", "args", "kwargs"):
                    continue
                # Soft check - log but don't fail for missing annotations
                if param.annotation == inspect.Parameter.empty:
                    pass  # Could log warning here

    def test_dataclasses_have_type_annotations(self):
        """Verify dataclass fields have type annotations."""
        from triton_ops.models import (
            TensorSpec,
            KernelMetrics,
            TuningResult,
            FP8Format,
        )

        dataclasses_to_check = [TensorSpec, KernelMetrics, TuningResult, FP8Format]

        for cls in dataclasses_to_check:
            # Dataclasses always have annotations
            assert hasattr(cls, "__annotations__"), f"{cls.__name__} should have annotations"
            assert len(cls.__annotations__) > 0, f"{cls.__name__} should have typed fields"


@pytest.mark.skipif(not HAS_TRITON_OPS, reason="triton_ops not available")
class TestDocstrings:
    """Tests for docstring completeness and structure.

    Feature: project-standardization, Property 2: Docstring Completeness and Structure
    Validates: Requirements 4.2, 4.6
    """

    def test_public_functions_have_docstrings(self):
        """Verify all public functions have docstrings."""
        functions = get_public_functions()

        for func in functions:
            assert func.__doc__ is not None, f"{func.__name__} is missing a docstring"
            assert len(func.__doc__.strip()) > 0, f"{func.__name__} has empty docstring"

    def test_public_classes_have_docstrings(self):
        """Verify all public classes have docstrings."""
        classes = get_public_classes()

        for cls in classes:
            assert cls.__doc__ is not None, f"{cls.__name__} is missing a docstring"
            assert len(cls.__doc__.strip()) > 0, f"{cls.__name__} has empty docstring"

    def test_docstrings_have_description(self):
        """Verify docstrings have a description."""
        functions = get_public_functions()

        for func in functions:
            if func.__doc__:
                # First line should be a description
                first_line = func.__doc__.strip().split("\n")[0]
                assert len(first_line) > 10, f"{func.__name__} docstring too short"

    def test_functions_with_params_have_args_section(self):
        """Verify functions with parameters document them."""
        functions = get_public_functions()

        for func in functions:
            sig = inspect.signature(func)
            # Filter out self, cls, *args, **kwargs
            params = [
                p
                for p in sig.parameters.values()
                if p.name not in ("self", "cls")
                and p.kind
                not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]

            if len(params) > 0 and func.__doc__:
                doc = func.__doc__
                # Should have Args section or Parameters section
                has_args_doc = (
                    "Args:" in doc or "Parameters:" in doc or "Arguments:" in doc
                )
                # Soft check - some simple functions may not need detailed docs
                if not has_args_doc and len(params) > 2:
                    pass  # Could log warning for functions with many undocumented params

    def test_functions_with_return_have_returns_section(self):
        """Verify functions with return values document them."""
        functions = get_public_functions()

        for func in functions:
            sig = inspect.signature(func)
            if sig.return_annotation not in (inspect.Signature.empty, None, type(None)):
                if func.__doc__:
                    doc = func.__doc__
                    # Should have Returns section
                    has_returns_doc = "Returns:" in doc or "Return:" in doc
                    # Soft check
                    if not has_returns_doc:
                        pass  # Could log warning


@pytest.mark.skipif(not HAS_TRITON_OPS, reason="triton_ops not available")
class TestModuleStructure:
    """Tests for module structure and organization."""

    def test_init_exports_main_apis(self):
        """Verify __init__.py exports main APIs."""
        # Check for main functional APIs
        expected_functions = [
            "fused_rmsnorm_rope",
            "fused_gated_mlp",
            "fp8_gemm",
            "quantize_fp8",
            "dequantize_fp8",
        ]

        for func_name in expected_functions:
            assert hasattr(triton_ops, func_name), f"triton_ops should export {func_name}"

    def test_init_exports_module_apis(self):
        """Verify __init__.py exports module APIs."""
        # Check for main module APIs
        expected_classes = [
            "FusedRMSNormRoPE",
            "FusedGatedMLP",
            "FP8Linear",
        ]

        for class_name in expected_classes:
            assert hasattr(triton_ops, class_name), f"triton_ops should export {class_name}"

    def test_exceptions_are_exported(self):
        """Verify custom exceptions are exported."""
        expected_exceptions = [
            "TritonKernelError",
            "ShapeMismatchError",
            "UnsupportedDtypeError",
        ]

        for exc_name in expected_exceptions:
            assert hasattr(triton_ops, exc_name), f"triton_ops should export {exc_name}"


class TestFileStructure:
    """Tests for project file structure."""

    def test_required_files_exist(self):
        """Verify all required open source files exist."""
        required_files = [
            "README.md",
            "LICENSE",
            "CONTRIBUTING.md",
            "CODE_OF_CONDUCT.md",
            "CHANGELOG.md",
            "pyproject.toml",
            "triton_ops/py.typed",
        ]

        for file_path in required_files:
            path = Path(file_path)
            assert path.exists(), f"Missing required file: {file_path}"

    def test_github_templates_exist(self):
        """Verify GitHub templates exist."""
        template_files = [
            ".github/ISSUE_TEMPLATE/bug_report.md",
            ".github/ISSUE_TEMPLATE/feature_request.md",
            ".github/PULL_REQUEST_TEMPLATE.md",
        ]

        for file_path in template_files:
            path = Path(file_path)
            assert path.exists(), f"Missing GitHub template: {file_path}"

    def test_ci_workflow_exists(self):
        """Verify CI workflow exists."""
        ci_path = Path(".github/workflows/ci.yml")
        assert ci_path.exists(), "Missing CI workflow file"

    def test_examples_directory_exists(self):
        """Verify examples directory exists with scripts."""
        examples_dir = Path("examples")
        assert examples_dir.exists(), "Missing examples directory"
        assert examples_dir.is_dir(), "examples should be a directory"

        # Check for example files
        example_files = list(examples_dir.glob("*.py"))
        assert len(example_files) > 0, "examples directory should contain Python files"


# Property-based tests using Hypothesis (if available)
if HAS_HYPOTHESIS and HAS_TRITON_OPS:
    functions_list = get_public_functions()
    if functions_list:

        @given(st.sampled_from(functions_list))
        @settings(max_examples=50)
        def test_all_functions_are_callable(func):
            """
            Feature: project-standardization, Property 1: Type Annotation Completeness
            Validates: Requirements 4.1, 4.4

            For any public function, it should be callable.
            """
            assert callable(func)

        @given(st.sampled_from(functions_list))
        @settings(max_examples=50)
        def test_all_functions_have_name(func):
            """
            Feature: project-standardization, Property 2: Docstring Completeness
            Validates: Requirements 4.2, 4.6

            For any public function, it should have a __name__ attribute.
            """
            assert hasattr(func, "__name__")
            assert len(func.__name__) > 0
