"""
Action history plugin: injects previous turns' function calls (from conversation.history_calls).
Reads only conversation; does not touch execution.
"""
from __future__ import annotations

from typing import Any, Dict, List


class ActionHistoryPlugin:
    """Builds action_history_section from conversation["history_calls"] (set by adapter from execution)."""

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
        action_history_section = ""
        if turn_index > 0:
            history_calls: List[List[str]] = state["conversation"].get("history_calls", [])
            history_lines: List[str] = []
            for idx in range(min(turn_index, len(history_calls))):
                if not history_calls[idx]:
                    continue
                joined = "; ".join(history_calls[idx])
                history_lines.append(f"- Turn {idx}: {joined}")
            if history_lines:
                action_history_section = "\nAction history so far:\n" + "\n".join(history_lines) + "\n"
        return {
            "action_history_section": action_history_section,
            "tool_result_memory_section": "",
        }
