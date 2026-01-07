"""Correctness verification utilities for benchmarking."""

from typing import Tuple

import torch


class CorrectnessVerifier:
    """Verifier for numerical correctness of kernel outputs.

    Compares kernel outputs against reference implementations
    with configurable tolerances.

    Args:
        rtol: Relative tolerance
        atol: Absolute tolerance
    """

    def __init__(
        self,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ):
        self.rtol = rtol
        self.atol = atol

    def verify(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
    ) -> Tuple[bool, dict]:
        """Verify that actual output matches expected within tolerance.

        Args:
            actual: Output from kernel under test
            expected: Output from reference implementation

        Returns:
            Tuple of (is_correct, details_dict)
        """
        # Ensure same device and dtype for comparison
        if actual.device != expected.device:
            expected = expected.to(actual.device)

        # Compute differences
        abs_diff = (actual.float() - expected.float()).abs()
        rel_diff = abs_diff / (expected.float().abs() + 1e-10)

        # Check tolerance
        within_atol = abs_diff <= self.atol
        within_rtol = rel_diff <= self.rtol
        within_tolerance = within_atol | within_rtol

        is_correct = within_tolerance.all().item()

        # Compute statistics
        details = {
            "is_correct": is_correct,
            "max_abs_diff": abs_diff.max().item(),
            "mean_abs_diff": abs_diff.mean().item(),
            "max_rel_diff": rel_diff.max().item(),
            "mean_rel_diff": rel_diff.mean().item(),
            "num_violations": (~within_tolerance).sum().item(),
            "total_elements": actual.numel(),
            "rtol": self.rtol,
            "atol": self.atol,
        }

        return is_correct, details

    def verify_allclose(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
    ) -> bool:
        """Simple allclose check.

        Args:
            actual: Output from kernel under test
            expected: Output from reference implementation

        Returns:
            True if outputs are close within tolerance
        """
        return torch.allclose(
            actual.float(),
            expected.float(),
            rtol=self.rtol,
            atol=self.atol,
        )

    def compute_relative_error(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
    ) -> float:
        """Compute maximum relative error.

        Args:
            actual: Output from kernel under test
            expected: Output from reference implementation

        Returns:
            Maximum relative error as a fraction
        """
        abs_diff = (actual.float() - expected.float()).abs()
        rel_diff = abs_diff / (expected.float().abs() + 1e-10)
        return rel_diff.max().item()


def verify_fp8_accuracy(
    fp8_result: torch.Tensor,
    fp16_baseline: torch.Tensor,
    max_relative_error: float = 0.01,
) -> Tuple[bool, dict]:
    """Verify FP8 result accuracy against FP16 baseline.

    Args:
        fp8_result: Result from FP8 computation
        fp16_baseline: Result from FP16 computation
        max_relative_error: Maximum allowed relative error (default 1%)

    Returns:
        Tuple of (is_within_tolerance, details_dict)
    """
    # Compute relative error
    abs_diff = (fp8_result.float() - fp16_baseline.float()).abs()
    baseline_abs = fp16_baseline.float().abs()

    # Avoid division by zero
    rel_error = abs_diff / (baseline_abs + 1e-10)

    # Check if within tolerance
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()

    is_within_tolerance = max_rel_error <= max_relative_error

    details = {
        "is_within_tolerance": is_within_tolerance,
        "max_relative_error": max_rel_error,
        "mean_relative_error": mean_rel_error,
        "tolerance": max_relative_error,
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
    }

    return is_within_tolerance, details


def verify_nan_inf_propagation(
    output: torch.Tensor,
    input_has_nan: bool,
    input_has_inf: bool,
) -> Tuple[bool, dict]:
    """Verify that NaN/Inf values are properly propagated.

    Args:
        output: Kernel output
        input_has_nan: Whether input contained NaN
        input_has_inf: Whether input contained Inf

    Returns:
        Tuple of (is_correct, details_dict)
    """
    output_has_nan = torch.isnan(output).any().item()
    output_has_inf = torch.isinf(output).any().item()

    # NaN should propagate
    nan_propagated = not input_has_nan or output_has_nan

    # Inf should propagate (may become NaN in some operations)
    inf_propagated = not input_has_inf or (output_has_inf or output_has_nan)

    is_correct = nan_propagated and inf_propagated

    details = {
        "is_correct": is_correct,
        "input_has_nan": input_has_nan,
        "input_has_inf": input_has_inf,
        "output_has_nan": output_has_nan,
        "output_has_inf": output_has_inf,
        "nan_propagated": nan_propagated,
        "inf_propagated": inf_propagated,
    }

    return is_correct, details
