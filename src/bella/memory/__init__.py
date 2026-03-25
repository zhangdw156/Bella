"""
Unified memory plugin layer for multi-turn inference.

Supports **single** and **composite** modes via environment variable::

    # single plugin (backward compatible)
    BELLA_MULTI_TURN_MEMORY_MODE=action_history

    # combine short-term + long-term with comma
    BELLA_MULTI_TURN_MEMORY_MODE=action_history,mem0

Available plugins (auto-registered via ``@register_memory``):

  Short-term (per-entry):
    none | action_history | tool_result_memory | tool_result_memory_v2

  Long-term (cross-entry, persistent):
    mem0
"""
from __future__ import annotations

import os

from bella.memory.base import MemoryComposite, MemoryPlugin, NoOpMemory
from bella.memory.registry import create_plugin, list_plugins, register_memory

# Import subpackages to trigger @register_memory decorators
import bella.memory.short_term  # noqa: F401
import bella.memory.long_term   # noqa: F401


def create_memory(mode: str | None = None) -> MemoryPlugin:
    """Create memory plugin(s) for the given mode string.

    *mode* may be a single name (``"action_history"``) or comma-separated
    names (``"action_history,mem0"``).  Returns a single plugin or a
    ``MemoryComposite`` that delegates to all specified plugins.
    """
    if mode is None:
        mode = os.getenv("BELLA_MULTI_TURN_MEMORY_MODE", "none")
    mode = (mode or "none").strip().lower()

    parts = [p.strip() for p in mode.split(",") if p.strip()]
    if not parts or parts == ["none"]:
        return NoOpMemory()

    plugins = [create_plugin(name) for name in parts]
    if len(plugins) == 1:
        return plugins[0]
    return MemoryComposite(plugins)


# Backward-compatible alias used by multi_turn_base adapter
get_plugin = create_memory


__all__ = [
    "MemoryPlugin",
    "MemoryComposite",
    "NoOpMemory",
    "create_memory",
    "get_plugin",
    "create_plugin",
    "register_memory",
    "list_plugins",
]
