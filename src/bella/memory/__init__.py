"""
Unified memory plugin layer for multi_turn_base.

Switch via BELLA_MULTI_TURN_MEMORY_MODE:
  none | action_history | tool_result_memory | tool_result_memory_v2
"""
from __future__ import annotations

import os
from bella.memory.base import MemoryPlugin, NoOpMemory
from bella.memory.plugins import (
    ActionHistoryPlugin,
    ToolResultMemoryPlugin,
    ToolResultMemoryV2Plugin,
)

def get_plugin(mode: str | None = None) -> MemoryPlugin:
    """Return the memory plugin for the given mode (default from env BELLA_MULTI_TURN_MEMORY_MODE)."""
    if mode is None:
        mode = os.getenv("BELLA_MULTI_TURN_MEMORY_MODE", "none")
    mode = (mode or "none").strip().lower()
    if mode == "none":
        return NoOpMemory()
    if mode == "action_history":
        return ActionHistoryPlugin()
    if mode == "tool_result_memory":
        return ToolResultMemoryPlugin()
    if mode == "tool_result_memory_v2":
        return ToolResultMemoryV2Plugin()
    return NoOpMemory()


__all__ = [
    "MemoryPlugin",
    "NoOpMemory",
    "get_plugin",
    "ActionHistoryPlugin",
    "ToolResultMemoryPlugin",
    "ToolResultMemoryV2Plugin",
]
