"""
Memory plugin base classes.

Core interface (general-purpose, per Mem0 paper paradigm):
  ``add(content)``   – ingest information into memory
  ``search(query)``  – retrieve relevant memories

Legacy BFCL hooks (wrappers that delegate to add/search):
  ``on_tool_result``     – called after tool execution → delegates to ``add``
  ``build_prompt_blocks`` – called before prompt build → delegates to ``search``

Lifecycle (managed by runner → adapter):
  ``open(session_id)``  →  inference loop  →  ``close()``

Composition:
  ``MemoryComposite`` – delegates all methods to a list of child plugins.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from bella.memory.observation import truncate_tool_output


class MemoryPlugin(ABC):
    """Abstract base for memory plugins.

    Subclasses must implement ``add`` and ``search``.  The legacy BFCL hooks
    (``on_tool_result`` / ``build_prompt_blocks``) have default implementations
    that delegate to ``add`` / ``search``, so short-term plugins that don't use
    the general interface can override them directly instead.
    """

    # ── lifecycle ────────────────────────────────────────────────────

    def open(self, session_id: str) -> None:
        """Called once before inference starts.  Create / reset the store."""

    def close(self) -> None:
        """Called once after inference ends.  Release resources, clear memory."""

    def init_state(self, conversation: Dict[str, Any]) -> None:
        """Called once per entry; may initialise keys on *conversation*."""

    # ── core interface (general-purpose) ─────────────────────────────

    @abstractmethod
    def add(self, content: str, metadata: Dict[str, Any] | None = None) -> None:
        """Ingest *content* into memory (extract facts, embed, store)."""

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[str]:
        """Retrieve up to *limit* memories relevant to *query*."""

    # ── legacy BFCL hooks (default: delegate to add/search) ─────────

    def on_tool_result(
        self,
        conversation: Dict[str, Any],
        turn_index: int,
        tool_call: str,
        tool_result_raw: str,
    ) -> None:
        """Called after tool execution.  Default: ``add()`` the formatted result."""
        truncated = truncate_tool_output(tool_result_raw, 800)
        self.add(f"Called {tool_call}. Result: {truncated}")

    def build_prompt_blocks(
        self,
        entry: Dict[str, Any],
        state: Dict[str, Any],
        turn_index: int,
    ) -> Dict[str, str]:
        """Build prompt injection blocks.  Default: ``search()`` with current turn text."""
        empty = {"action_history_section": "", "tool_result_memory_section": ""}
        if turn_index == 0:
            return empty

        conversation = state.get("conversation", {})
        turn_texts: List[str] = conversation.get("turn_texts", [])
        query = turn_texts[turn_index] if turn_index < len(turn_texts) else ""
        if not query:
            return empty

        memories = self.search(query, limit=5)
        if not memories:
            return empty

        lines = [f"- {m}" for m in memories]
        section = (
            "\nRelevant memories from past interactions:\n"
            + "\n".join(lines)
            + "\n"
        )
        return {"action_history_section": "", "tool_result_memory_section": section}


class NoOpMemory(MemoryPlugin):
    """Memory plugin that does nothing."""

    def add(self, content: str, metadata: Dict[str, Any] | None = None) -> None:
        pass

    def search(self, query: str, limit: int = 5) -> List[str]:
        return []

    def on_tool_result(self, conversation: Dict[str, Any], turn_index: int, tool_call: str, tool_result_raw: str) -> None:
        pass

    def build_prompt_blocks(self, entry: Dict[str, Any], state: Dict[str, Any], turn_index: int) -> Dict[str, str]:
        return {"action_history_section": "", "tool_result_memory_section": ""}


class MemoryComposite(MemoryPlugin):
    """Composes multiple plugins, delegating all methods to every child."""

    def __init__(self, plugins: List[MemoryPlugin]) -> None:
        self._plugins = list(plugins)

    def open(self, session_id: str) -> None:
        for p in self._plugins:
            p.open(session_id)

    def close(self) -> None:
        for p in self._plugins:
            p.close()

    def init_state(self, conversation: Dict[str, Any]) -> None:
        for p in self._plugins:
            p.init_state(conversation)

    def add(self, content: str, metadata: Dict[str, Any] | None = None) -> None:
        for p in self._plugins:
            p.add(content, metadata)

    def search(self, query: str, limit: int = 5) -> List[str]:
        results: List[str] = []
        for p in self._plugins:
            results.extend(p.search(query, limit))
        return results[:limit]

    def on_tool_result(self, conversation: Dict[str, Any], turn_index: int, tool_call: str, tool_result_raw: str) -> None:
        for p in self._plugins:
            p.on_tool_result(conversation, turn_index, tool_call, tool_result_raw)

    def build_prompt_blocks(self, entry: Dict[str, Any], state: Dict[str, Any], turn_index: int) -> Dict[str, str]:
        merged: Dict[str, str] = {"action_history_section": "", "tool_result_memory_section": ""}
        for p in self._plugins:
            blocks = p.build_prompt_blocks(entry, state, turn_index)
            for key, value in blocks.items():
                if value:
                    merged[key] = merged.get(key, "") + value
        return merged
