"""Common utilities for Triton operators."""

import torch


def sync_cuda() -> None:
    """Synchronize CUDA if available.

    This is a no-op on CPU-only systems.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
