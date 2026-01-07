"""Configuration cache for auto-tuning results."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class ConfigCache:
    """Cache for storing and retrieving optimal kernel configurations.

    Configurations are cached based on:
    - Problem size (dimensions)
    - Device name
    - Kernel type

    Args:
        cache_dir: Directory to store cache files. If None, uses in-memory cache only.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory_cache: Dict[str, Dict[str, Any]] = {}

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(
        self,
        kernel_type: str,
        problem_size: Tuple[int, ...],
        device: str,
    ) -> str:
        """Create a unique cache key.

        Args:
            kernel_type: Type of kernel
            problem_size: Tuple of problem dimensions
            device: Device name

        Returns:
            Unique string key
        """
        key_data = f"{kernel_type}_{problem_size}_{device}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(
        self,
        kernel_type: str,
        problem_size: Tuple[int, ...],
        device: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve cached configuration.

        Args:
            kernel_type: Type of kernel
            problem_size: Tuple of problem dimensions
            device: Device name

        Returns:
            Cached configuration or None if not found
        """
        key = self._make_key(kernel_type, problem_size, device)

        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key].copy()

        # Check file cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    with open(cache_file) as f:
                        config = json.load(f)
                    # Store in memory cache for faster access
                    self._memory_cache[key] = config
                    return config.copy()
                except (OSError, json.JSONDecodeError):
                    pass

        return None

    def set(
        self,
        kernel_type: str,
        problem_size: Tuple[int, ...],
        device: str,
        config: Dict[str, Any],
    ) -> None:
        """Store configuration in cache.

        Args:
            kernel_type: Type of kernel
            problem_size: Tuple of problem dimensions
            device: Device name
            config: Configuration to cache
        """
        key = self._make_key(kernel_type, problem_size, device)

        # Store in memory cache
        self._memory_cache[key] = config.copy()

        # Store in file cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.json"
            try:
                with open(cache_file, "w") as f:
                    json.dump(config, f, indent=2)
            except OSError:
                pass  # Silently fail file writes

    def clear(self) -> None:
        """Clear all cached configurations."""
        self._memory_cache.clear()

        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                except OSError:
                    pass

    def get_all_keys(self) -> list:
        """Get all cached keys.

        Returns:
            List of cache keys
        """
        keys = set(self._memory_cache.keys())

        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                keys.add(cache_file.stem)

        return list(keys)

    def __contains__(self, item: Tuple[str, Tuple[int, ...], str]) -> bool:
        """Check if configuration is cached.

        Args:
            item: Tuple of (kernel_type, problem_size, device)

        Returns:
            True if cached
        """
        kernel_type, problem_size, device = item
        return self.get(kernel_type, problem_size, device) is not None

    def __len__(self) -> int:
        """Return number of cached configurations."""
        return len(self.get_all_keys())
