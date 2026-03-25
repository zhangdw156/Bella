"""
Tool result memory v2 plugin: stores factual observations (verbalized)
in conversation, injects "Turn t | call => observation" block.
Operates only on conversation.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from bella.memory.base import MemoryPlugin
from bella.memory.observation import observation_from_tool_result, truncate_tool_output
from bella.memory.registry import register_memory


@register_memory("tool_result_memory_v2")
class ToolResultMemoryV2Plugin(MemoryPlugin):
    """Writes to ``conversation["tool_result_memory_items"]`` with observation text; builds block with ``=>``."""

    def __init__(self) -> None:
        self.max_chars_per_item: int = int(
            os.getenv("BELLA_TOOL_RESULT_MEMORY_MAX_CHARS_PER_ITEM", "600")
        )
        self.max_items: int = int(os.getenv("BELLA_TOOL_RESULT_MEMORY_MAX_ITEMS", "3"))
        self.max_total_chars: int = int(
            os.getenv("BELLA_TOOL_RESULT_MEMORY_MAX_TOTAL_CHARS", "1800")
        )

    def init_state(self, conversation: Dict[str, Any]) -> None:
        conversation["tool_result_memory_items"] = []

    def add(self, content: str, metadata: Dict[str, Any] | None = None) -> None:
        pass

    def search(self, query: str, limit: int = 5) -> List[str]:
        return []

    def on_tool_result(
        self,
        conversation: Dict[str, Any],
        turn_index: int,
        tool_call: str,
        tool_result_raw: str,
    ) -> None:
        truncated = truncate_tool_output(tool_result_raw, self.max_chars_per_item)
        observation = observation_from_tool_result(tool_call, truncated)
        items: List[Dict[str, Any]] = conversation.get("tool_result_memory_items", [])
        items.append(
            {
                "turn": turn_index,
                "tool_call": tool_call,
                "tool_result": observation,
            }
        )
        conversation["tool_result_memory_items"] = items

    def _build_block(self, conversation: Dict[str, Any], current_turn_index: int) -> str:
        items: List[Dict[str, Any]] = conversation.get("tool_result_memory_items", [])
        if not items:
            return ""
        eligible = [it for it in items if int(it.get("turn", -1)) < current_turn_index]
        if not eligible:
            return ""
        if self.max_items > 0:
            eligible = eligible[-self.max_items :]
        lines: List[str] = []
        for it in eligible:
            t = it.get("turn", "?")
            call = str(it.get("tool_call", ""))
            result = str(it.get("tool_result", ""))
            lines.append(f"- Turn {t} | {call} => {result}")
        if self.max_total_chars > 0:
            while lines and len("\n".join(lines)) > self.max_total_chars:
                lines.pop(0)
        return "\n".join(lines)

    def build_prompt_blocks(
        self,
        entry: Dict[str, Any],
        state: Dict[str, Any],
        turn_index: int,
    ) -> Dict[str, str]:
        tool_result_memory_section = ""
        if turn_index > 0:
            conversation = state["conversation"]
            memory_block = self._build_block(conversation, turn_index)
            if memory_block:
                tool_result_memory_section = (
                    "\nTool result observations so far:\n"
                    + memory_block
                    + "\n"
                )
        return {
            "action_history_section": "",
            "tool_result_memory_section": tool_result_memory_section,
        }

    def debug_tool_memory_inner(self, state: Dict[str, Any], turn_index: int) -> str:
        """Same inner lines as legacy ``_build_tool_result_memory_block``; for debug parity."""
        return self._build_block(state["conversation"], turn_index)
