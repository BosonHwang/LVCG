"""Dataset providers for LVCG linear probing."""

from __future__ import annotations

from typing import Dict, Type

from .aireadi.provider import AIREADIConditionProvider
from .base import DatasetBundle, DatasetProvider
from .benchmark_provider import BenchmarkProvider

DATASET_REGISTRY: Dict[str, Type[DatasetProvider]] = {
    "benchmark": BenchmarkProvider,
    "aireadi_condition": AIREADIConditionProvider,
}

_RESERVED_KEYS = {"type"}


def create_provider(name: str, config: dict) -> DatasetProvider:
    provider_type = config.get("type")
    if provider_type is None:
        raise ValueError(f"Dataset '{name}' is missing required 'type' key")
    if provider_type not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset type: {provider_type}. "
            f"Available: {sorted(DATASET_REGISTRY)}"
        )
    cls = DATASET_REGISTRY[provider_type]
    kwargs = {k: v for k, v in config.items() if k not in _RESERVED_KEYS}
    provider = cls(**kwargs)
    provider.name = name
    return provider


__all__ = [
    "DATASET_REGISTRY",
    "DatasetBundle",
    "DatasetProvider",
    "BenchmarkProvider",
    "AIREADIConditionProvider",
    "create_provider",
]
