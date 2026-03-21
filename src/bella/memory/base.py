"""
Unified memory plugin interface for multi-turn inference.

Plugins operate only on conversation state (read/write); they do not touch
execution state. The adapter syncs execution -> conversation (e.g. history_calls)
and calls plugin hooks; plugins produce prompt blocks for injection.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class MemoryPlugin(Protocol):
    """Minimal memory plugin interface for multi_turn_base."""

    def init_state(self, conversation: Dict[str, Any]) -> None:
        """Called once per entry; plugin may init keys on conversation (e.g. tool_result_memory_items)."""
        ...

    def on_tool_result(
        self,
        conversation: Dict[str, Any],
        turn_index: int,
        tool_call: str,
        tool_result_raw: str,
    ) -> None:
        """Called after each tool execution; plugin may append to conversation memory state."""
        ...

    def build_prompt_blocks(
        self,
        entry: Dict[str, Any],
        state: Dict[str, Any],
        turn_index: int,
    ) -> Dict[str, str]:
        """Return placeholder name -> content for user prompt, e.g. action_history_section, tool_result_memory_section."""
        ...


class NoOpMemory:
    """Memory plugin that injects nothing. Used for mode=none."""

    def init_state(self, conversation: Dict[str, Any]) -> None:
        pass

    def on_tool_result(
        self,
        conversation: Dict[str, Any],
        turn_index: int,
        tool_call: str,
        tool_result_raw: str,
    ) -> None:
        pass

    def build_prompt_blocks(
        self,
        entry: Dict[str, Any],
        state: Dict[str, Any],
        turn_index: int,
    ) -> Dict[str, str]:
        return {
            "action_history_section": "",
            "tool_result_memory_section": "",
        }
