"""Pytest configuration and fixtures for Triton operators tests."""

import pytest
import torch
from hypothesis import Verbosity, settings

# Configure hypothesis for property-based testing
settings.register_profile("ci", max_examples=100, deadline=None)
settings.register_profile("dev", max_examples=20, deadline=None)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose, deadline=None)
settings.load_profile("dev")


@pytest.fixture
def device():
    """Return the device to use for tests."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def cuda_device():
    """Return CUDA device, skip if not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture
def small_hidden_dim():
    """Return small hidden dimension for quick tests."""
    return 256


@pytest.fixture
def medium_hidden_dim():
    """Return medium hidden dimension."""
    return 2048


@pytest.fixture
def large_hidden_dim():
    """Return large hidden dimension."""
    return 4096
