"""Property-based tests for Auto-Tuner.

Feature: triton-fused-operators
Tests Property 7: Auto-Tuner Cache Consistency
Validates: Requirements 4.5
"""

import tempfile

import pytest

# Skip GPU-specific tests if CUDA is not available
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class TestAutoTunerCacheConsistency:
    """Property tests for auto-tuner cache consistency."""

    @given(
        M=st.integers(min_value=64, max_value=512),
        N=st.integers(min_value=64, max_value=512),
        K=st.integers(min_value=64, max_value=512),
    )
    @settings(max_examples=100, deadline=None)
    def test_cache_consistency(self, M, N, K):
        """
        Feature: triton-fused-operators, Property 7: Auto-Tuner Cache Consistency
        Validates: Requirements 4.5

        For any problem size and device configuration, if the auto-tuner has
        previously found an optimal configuration, retrieving the cached
        configuration should return the same configuration that was stored.
        """
        from triton_ops.autotuner.cache import ConfigCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ConfigCache(cache_dir=tmpdir)

            kernel_type = "test_kernel"
            problem_size = (M, N, K)
            device = "cuda:0"

            # Create a configuration
            config = {
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_K": 32,
                "num_warps": 4,
            }

            # Store configuration
            cache.set(kernel_type, problem_size, device, config)

            # Retrieve configuration
            retrieved = cache.get(kernel_type, problem_size, device)

            # Verify consistency
            assert retrieved is not None, "Cached config should be retrievable"
            assert retrieved == config, "Retrieved config should match stored config"

    @given(
        num_configs=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_multiple_configs(self, num_configs):
        """Test caching multiple configurations."""
        from triton_ops.autotuner.cache import ConfigCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ConfigCache(cache_dir=tmpdir)

            # Store multiple configs
            configs = {}
            for i in range(num_configs):
                problem_size = (64 * (i + 1), 64 * (i + 1), 32)
                config = {"BLOCK_M": 64, "index": i}
                cache.set("test", problem_size, "cuda:0", config)
                configs[problem_size] = config

            # Verify all can be retrieved
            for problem_size, expected_config in configs.items():
                retrieved = cache.get("test", problem_size, "cuda:0")
                assert retrieved == expected_config


class TestAutoTunerConfigSpace:
    """Unit tests for configuration space."""

    def test_config_generation(self):
        """
        Validates: Requirements 4.1, 4.2, 4.3

        THE Auto_Tuning framework SHALL search over BLOCK_SIZE, num_warps,
        and num_stages parameters.
        """
        from triton_ops.autotuner.configs import (
            RMSNORM_ROPE_CONFIGS,
            generate_configs,
        )

        configs = generate_configs(RMSNORM_ROPE_CONFIGS)

        # Verify all parameter combinations are generated
        assert len(configs) > 0

        # Verify each config has all required keys
        for config in configs:
            assert "BLOCK_SIZE" in config
            assert "num_warps" in config
            assert "num_stages" in config

        # Verify parameter values are from config space
        block_sizes = set(c["BLOCK_SIZE"] for c in configs)
        assert block_sizes == set(RMSNORM_ROPE_CONFIGS["BLOCK_SIZE"])

    def test_default_configs(self):
        """Test default configuration retrieval."""
        from triton_ops.autotuner.configs import get_default_config

        # Test each kernel type
        for kernel_type in ["rmsnorm_rope", "gated_mlp", "fp8_gemm"]:
            config = get_default_config(kernel_type)
            assert config is not None
            assert len(config) > 0


class TestAutoTunerIntegration:
    """Integration tests for auto-tuner."""

    def test_tuner_basic(self):
        """Test basic tuner functionality."""
        from triton_ops.autotuner.tuner import TritonAutoTuner

        # Simple kernel function for testing
        def dummy_kernel(*args, BLOCK_SIZE=64, num_warps=4, **kwargs):
            # Just create and return a tensor
            return torch.randn(100, device="cuda")

        config_space = {
            "BLOCK_SIZE": [64, 128],
            "num_warps": [4],
        }

        tuner = TritonAutoTuner(
            kernel_fn=dummy_kernel,
            config_space=config_space,
            warmup_runs=2,
            benchmark_runs=5,
        )

        # Run tuning
        result = tuner.tune(
            problem_size=(100,),
            device="cuda:0",
            kernel_type="test",
        )

        assert result.best_config is not None
        assert result.metrics is not None
        assert result.metrics.latency_ms > 0
