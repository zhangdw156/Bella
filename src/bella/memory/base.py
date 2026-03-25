"""
Memory plugin base classes.

* ``MemoryPlugin``  – abstract base for all plugins (short-term & long-term).
* ``NoOpMemory``    – concrete no-op fallback.
* ``MemoryComposite`` – composes multiple plugins, merging their prompt blocks.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class MemoryPlugin(ABC):
    """Abstract base for multi-turn memory plugins.

    Plugins operate only on conversation state (read/write); they do not touch
    execution state.  The adapter syncs execution → conversation and calls
    plugin hooks; plugins produce prompt blocks for injection.
    """

    @abstractmethod
    def init_state(self, conversation: Dict[str, Any]) -> None:
        """Called once per entry; may initialise keys on *conversation*."""

    @abstractmethod
    def on_tool_result(
        self,
        conversation: Dict[str, Any],
        turn_index: int,
        tool_call: str,
        tool_result_raw: str,
    ) -> None:
        """Called after each tool execution; may update conversation memory state."""

    @abstractmethod
    def build_prompt_blocks(
        self,
        entry: Dict[str, Any],
        state: Dict[str, Any],
        turn_index: int,
    ) -> Dict[str, str]:
        """Return ``{placeholder_name: content}`` for user-prompt injection.

        Standard keys: ``action_history_section``, ``tool_result_memory_section``.
        """


class NoOpMemory(MemoryPlugin):
    """Memory plugin that injects nothing.  Used when mode is ``none``."""

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


class MemoryComposite(MemoryPlugin):
    """Composes multiple ``MemoryPlugin`` instances.

    * ``init_state`` / ``on_tool_result`` — delegates to every child.
    * ``build_prompt_blocks`` — merges outputs by concatenating values per key
      (so short-term + long-term blocks can coexist in the same prompt).
    """

    def __init__(self, plugins: List[MemoryPlugin]) -> None:
        self._plugins = list(plugins)

    def init_state(self, conversation: Dict[str, Any]) -> None:
        for p in self._plugins:
            p.init_state(conversation)

    def on_tool_result(
        self,
        conversation: Dict[str, Any],
        turn_index: int,
        tool_call: str,
        tool_result_raw: str,
    ) -> None:
        for p in self._plugins:
            p.on_tool_result(conversation, turn_index, tool_call, tool_result_raw)

    def build_prompt_blocks(
        self,
        entry: Dict[str, Any],
        state: Dict[str, Any],
        turn_index: int,
    ) -> Dict[str, str]:
        merged: Dict[str, str] = {
            "action_history_section": "",
            "tool_result_memory_section": "",
        }
        for p in self._plugins:
            blocks = p.build_prompt_blocks(entry, state, turn_index)
            for key, value in blocks.items():
                if value:
                    merged[key] = merged.get(key, "") + value
        return merged
