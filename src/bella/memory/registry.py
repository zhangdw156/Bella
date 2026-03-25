"""
Memory plugin registration and lookup.

Plugins self-register via the ``@register_memory`` decorator; the factory
``create_plugin`` instantiates them by name at runtime.
"""
from __future__ import annotations

import logging
from typing import Dict, Type

from bella.memory.base import MemoryPlugin, NoOpMemory

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, Type[MemoryPlugin]] = {}


def register_memory(name: str):
    """Class decorator that registers a MemoryPlugin implementation under *name*."""
    def decorator(cls: Type[MemoryPlugin]) -> Type[MemoryPlugin]:
        key = name.strip().lower()
        if key in _REGISTRY:
            logger.warning(
                "memory plugin %r re-registered: %s -> %s",
                key,
                _REGISTRY[key].__name__,
                cls.__name__,
            )
        _REGISTRY[key] = cls
        return cls
    return decorator


def create_plugin(name: str) -> MemoryPlugin:
    """Instantiate a registered plugin by name.  Raises on unknown name."""
    key = name.strip().lower()
    if key in ("none", ""):
        return NoOpMemory()
    cls = _REGISTRY.get(key)
    if cls is None:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none registered)"
        raise ValueError(
            f"Unknown memory plugin: {name!r}. Available: {available}"
        )
    return cls()


def list_plugins() -> list[str]:
    """Return sorted list of registered plugin names."""
    return sorted(_REGISTRY.keys())
