"""
Benchmark registry.

Benchmarks self-register via ``@register_benchmark``; the general runner
looks them up with ``get_benchmark``.

Sub-packages are imported at the bottom to trigger registration decorators.
"""
from __future__ import annotations

import logging
from typing import Dict, Type

from bella.benchmarks.base import Benchmark, InferAdapter

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, Type[Benchmark]] = {}


def register_benchmark(name: str):
    """Class decorator that registers a ``Benchmark`` implementation."""
    def decorator(cls: Type[Benchmark]) -> Type[Benchmark]:
        key = name.strip().lower()
        if key in _REGISTRY:
            logger.warning(
                "benchmark %r re-registered: %s -> %s",
                key, _REGISTRY[key].__name__, cls.__name__,
            )
        _REGISTRY[key] = cls
        return cls
    return decorator


def get_benchmark(name: str) -> Benchmark:
    """Instantiate a registered benchmark by name."""
    key = name.strip().lower()
    cls = _REGISTRY.get(key)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise ValueError(f"Unknown benchmark: {name!r}. Available: {available}")
    return cls()


def list_benchmarks() -> list[str]:
    """Return sorted list of registered benchmark names."""
    return sorted(_REGISTRY.keys())


# Import sub-packages to trigger @register_benchmark decorators
import bella.benchmarks.bfcl  # noqa: E402, F401


__all__ = [
    "Benchmark",
    "InferAdapter",
    "register_benchmark",
    "get_benchmark",
    "list_benchmarks",
]
