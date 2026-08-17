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
        # 参考实现可能返回 NumPy ndarray（numpy 纯计算路径），先统一转 torch
        # 张量再比较，避免 ndarray 无 .float() 的兼容问题。
        def _as_tensor(t: torch.Tensor):
            if torch.is_tensor(t):
                t = t.float()
            else:
                t = torch.as_tensor(t, dtype=torch.float32)
            return t.to(actual.device)

        return bool(
            torch.allclose(
                _as_tensor(actual),
                _as_tensor(expected),
                rtol=self.rtol,
                atol=self.atol,
            )
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
        return float(rel_diff.max().item())


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
