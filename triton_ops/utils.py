"""Common utilities for Triton operators."""

import torch


def sync_cuda() -> None:
    """Synchronize CUDA if available.

    This is a no-op on CPU-only systems.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Activation type constants
ACTIVATION_SILU = "silu"
ACTIVATION_GELU = "gelu"
VALID_ACTIVATIONS = (ACTIVATION_SILU, ACTIVATION_GELU)


# Minimum latency for metric calculations (avoids division by zero)
MIN_LATENCY_MS = 1e-9
