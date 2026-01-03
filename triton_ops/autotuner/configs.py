"""Configuration spaces for auto-tuning Triton kernels."""

from typing import Any, Dict, List
from itertools import product


# RMSNorm + RoPE configuration space
RMSNORM_ROPE_CONFIGS = {
    'BLOCK_SIZE': [64, 128, 256, 512, 1024],
    'num_warps': [2, 4, 8],
    'num_stages': [1, 2, 3],
}

# Gated MLP configuration space
GATED_MLP_CONFIGS = {
    'BLOCK_M': [32, 64, 128],
    'BLOCK_N': [32, 64, 128],
    'BLOCK_K': [32, 64],
    'num_warps': [4, 8],
    'num_stages': [2, 3, 4],
}

# FP8 GEMM configuration space
FP8_GEMM_CONFIGS = {
    'BLOCK_M': [64, 128, 256],
    'BLOCK_N': [64, 128, 256],
    'BLOCK_K': [32, 64],
    'GROUP_SIZE_M': [4, 8],
    'num_warps': [4, 8],
    'num_stages': [3, 4, 5],
}


def generate_configs(config_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all configurations from a configuration space.
    
    Args:
        config_space: Dictionary mapping parameter names to lists of values
        
    Returns:
        List of configuration dictionaries
    """
    keys = list(config_space.keys())
    values = list(config_space.values())
    
    configs = []
    for combo in product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)
    
    return configs


def filter_valid_configs(
    configs: List[Dict[str, Any]],
    hidden_dim: int = None,
    intermediate_dim: int = None,
    M: int = None,
    N: int = None,
    K: int = None,
) -> List[Dict[str, Any]]:
    """Filter configurations to only include valid ones for given dimensions.
    
    Args:
        configs: List of configurations
        hidden_dim: Hidden dimension (for RMSNorm/RoPE)
        intermediate_dim: Intermediate dimension (for Gated MLP)
        M, N, K: Matrix dimensions (for GEMM)
        
    Returns:
        Filtered list of valid configurations
    """
    valid = []
    
    for config in configs:
        is_valid = True
        
        # Check BLOCK_SIZE doesn't exceed hidden_dim
        if 'BLOCK_SIZE' in config and hidden_dim is not None:
            if config['BLOCK_SIZE'] > hidden_dim:
                is_valid = False
        
        # Check BLOCK_M doesn't exceed M
        if 'BLOCK_M' in config and M is not None:
            if config['BLOCK_M'] > M * 2:  # Allow some flexibility
                is_valid = False
        
        # Check BLOCK_N doesn't exceed N
        if 'BLOCK_N' in config and N is not None:
            if config['BLOCK_N'] > N * 2:
                is_valid = False
        
        # Check BLOCK_K doesn't exceed K
        if 'BLOCK_K' in config and K is not None:
            if config['BLOCK_K'] > K:
                is_valid = False
        
        if is_valid:
            valid.append(config)
    
    return valid


def get_default_config(kernel_type: str) -> Dict[str, Any]:
    """Get default configuration for a kernel type.
    
    Args:
        kernel_type: One of "rmsnorm_rope", "gated_mlp", "fp8_gemm"
        
    Returns:
        Default configuration dictionary
    """
    defaults = {
        'rmsnorm_rope': {
            'BLOCK_SIZE': 128,
            'num_warps': 4,
            'num_stages': 2,
        },
        'gated_mlp': {
            'BLOCK_M': 64,
            'BLOCK_N': 64,
            'BLOCK_K': 32,
            'num_warps': 4,
            'num_stages': 3,
        },
        'fp8_gemm': {
            'BLOCK_M': 128,
            'BLOCK_N': 128,
            'BLOCK_K': 32,
            'GROUP_SIZE_M': 8,
            'num_warps': 4,
            'num_stages': 4,
        },
    }
    
    return defaults.get(kernel_type, {})
