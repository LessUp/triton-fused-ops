"""Common utilities for Triton operators."""

import torch

# Minimum latency sentinel — avoids zero-division in derived metric calculations
MIN_LATENCY_MS: float = 1e-9


def sync_cuda() -> None:
    """Synchronize CUDA if available.

    This is a no-op on CPU-only systems.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
