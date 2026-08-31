"""Property-based tests for Benchmark Suite.

Feature: triton-fused-operators
Tests Property 8: Benchmark Correctness Verification
Validates: Requirements 5.3
"""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

# Skip GPU-specific tests if CUDA is not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TestBenchmarkCorrectnessVerification:
    """Property tests for benchmark correctness verification."""

    @given(
        size=st.integers(min_value=10, max_value=1000),
        noise_level=st.floats(min_value=0.0, max_value=0.001),
    )
    @settings(max_examples=100, deadline=None)
    def test_matching_results_identified_as_correct(self, size, noise_level):
        """
        Feature: triton-fused-operators, Property 8: Benchmark Correctness Verification
        Validates: Requirements 5.3

        For any kernel implementation and reference implementation, the benchmark
        suite's correctness verification should correctly identify matching results
        (within tolerance) as correct.
        """
        from trifuse.benchmark.correctness import CorrectnessVerifier

        # FP16 数据下 rtol=1e-3/atol=1e-5 过紧（FP16 ULP 即可超过），用 FP16 合理容差
        verifier = CorrectnessVerifier(rtol=1e-2, atol=1e-3)

        # Create matching tensors with small noise
        expected = torch.randn(size, device="cuda", dtype=torch.float16)
        actual = expected + torch.randn_like(expected) * noise_level

        is_correct, details = verifier.verify(actual, expected)

        # With small noise, should be identified as correct
        if noise_level < 1e-4:
            assert is_correct, f"Matching results should be identified as correct: {details}"

    @given(
        size=st.integers(min_value=10, max_value=1000),
        error_magnitude=st.floats(min_value=0.1, max_value=1.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_non_matching_results_identified_as_incorrect(self, size, error_magnitude):
        """
        Feature: triton-fused-operators, Property 8: Benchmark Correctness Verification
        Validates: Requirements 5.3

        The benchmark suite's correctness verification should correctly identify
        non-matching results (outside tolerance) as incorrect.
        """
        from trifuse.benchmark.correctness import CorrectnessVerifier

        verifier = CorrectnessVerifier(rtol=1e-3, atol=1e-5)

        # Create non-matching tensors with large error
        expected = torch.randn(size, device="cuda", dtype=torch.float16)
        actual = expected + torch.randn_like(expected) * error_magnitude

        is_correct, details = verifier.verify(actual, expected)

        # With large error, should be identified as incorrect
        assert not is_correct, f"Non-matching results should be identified as incorrect: {details}"


class TestNaNInfVerification:
    """Unit tests for NaN/Inf propagation verification."""

    def test_nan_propagation_verification(self):
        """Test NaN propagation verification."""
        from trifuse.benchmark.correctness import verify_nan_inf_propagation

        # Output with NaN when input had NaN
        output_with_nan = torch.tensor([1.0, float("nan"), 3.0], device="cuda")
        is_correct, details = verify_nan_inf_propagation(
            output_with_nan, input_has_nan=True, input_has_inf=False
        )
        assert is_correct, "NaN should be properly propagated"

        # Output without NaN when input had NaN (incorrect)
        output_without_nan = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        is_correct, details = verify_nan_inf_propagation(
            output_without_nan, input_has_nan=True, input_has_inf=False
        )
        assert not is_correct, "Missing NaN propagation should be detected"

    def test_inf_propagation_verification(self):
        """Test Inf propagation verification."""
        from trifuse.benchmark.correctness import verify_nan_inf_propagation

        # Output with Inf when input had Inf
        output_with_inf = torch.tensor([1.0, float("inf"), 3.0], device="cuda")
        is_correct, details = verify_nan_inf_propagation(
            output_with_inf, input_has_nan=False, input_has_inf=True
        )
        assert is_correct, "Inf should be properly propagated"


class TestBenchmarkSuiteIntegration:
    """Integration tests for benchmark suite."""

    def test_benchmark_suite_basic(self):
        """Test basic benchmark suite functionality."""
        from trifuse.benchmark.suite import BenchmarkSuite

        suite = BenchmarkSuite(warmup_runs=2, benchmark_runs=5)

        # Simple functions for testing
        def triton_fn(x):
            return x * 2

        def reference_fn(x):
            return x * 2

        x = torch.randn(100, device="cuda", dtype=torch.float16)

        result = suite.benchmark_kernel(
            triton_fn,
            reference_fn,
            "test_kernel",
            (100,),
            x,
        )

        assert result.correctness
        assert result.metrics.latency_ms > 0

    def test_report_generation(self):
        """Test report generation."""
        from trifuse.benchmark.suite import BenchmarkSuite

        suite = BenchmarkSuite(warmup_runs=2, benchmark_runs=5)

        # Add a simple benchmark
        def fn(x):
            return x * 2

        x = torch.randn(100, device="cuda", dtype=torch.float16)
        suite.benchmark_kernel(fn, fn, "test", (100,), x)

        # Generate reports
        text_report = suite.generate_report(format="text")
        json_report = suite.generate_report(format="json")

        assert "test" in text_report
        assert "test" in json_report
