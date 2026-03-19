from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ToolCall:
    """
    A single tool invocation extracted from an LLM response.
    """

    name: str
    arguments: Dict[str, Any]
    tool_call_id: Optional[str] = None


@dataclass(frozen=True)
class ToolResult:
    """
    The result of executing a single ToolCall in an environment.
    """

    name: str
    output: str
    tool_call_id: Optional[str] = None


class EnvironmentSession:
    """
    Minimal environment session interface for multi-turn tasks.
    """

    def execute_one(self, call: ToolCall) -> ToolResult:
        raise NotImplementedError

    def snapshot(self) -> Dict[str, Any]:
        """
        Optional: return a debug-friendly snapshot of the current environment state.
        """

        return {}

