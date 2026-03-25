from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from bella.env.base import ToolCall, ToolResult


def render_func_call_string(name: str, arguments: Dict[str, Any]) -> str:
    """
    Render a single function call string compatible with BFCL execution utils.
    Example: name='ls', arguments={'folder': 'document'} -> "ls(folder='document')"
    """
    if not arguments:
        return f"{name}()"
    parts = [f"{k}={repr(v)}" for k, v in arguments.items()]
    return f"{name}({','.join(parts)})"


def extract_tool_calls_from_openai_chat_response(response: Any) -> List[ToolCall]:
    """Extract all tool calls from an OpenAI /chat/completions response."""
    tool_calls = getattr(response.choices[0].message, "tool_calls", None)
    if not tool_calls:
        return []

    extracted: List[ToolCall] = []
    for first in tool_calls:
        func = getattr(first, "function", None)
        if not func:
            continue

        name = getattr(func, "name", None)
        if not isinstance(name, str) or not name:
            continue

        raw_args = getattr(func, "arguments", "") or ""
        args_obj: Dict[str, Any] = {}
        if raw_args:
            try:
                parsed = json.loads(raw_args)
                if isinstance(parsed, dict):
                    args_obj = parsed
            except Exception:
                # Best-effort: if parse fails, keep empty args to avoid crashing.
                args_obj = {}

        tool_call_id = getattr(first, "id", None)
        if tool_call_id is not None and not isinstance(tool_call_id, str):
            tool_call_id = str(tool_call_id)

        extracted.append(ToolCall(name=name, arguments=args_obj, tool_call_id=tool_call_id))

    return extracted


def extract_first_tool_call_from_openai_chat_response(response: Any) -> Optional[ToolCall]:
    """
    Extract at most one tool call from OpenAI /chat/completions response.
    """
    all_calls = extract_tool_calls_from_openai_chat_response(response)
    if not all_calls:
        return None
    return all_calls[0]


def execute_first_tool_call(
    *,
    env_session: Any,
    response: Any,
) -> Tuple[Optional[ToolCall], Optional[ToolResult]]:
    """
    Execute at most one tool call from the model response using the given env_session.
    """
    call = extract_first_tool_call_from_openai_chat_response(response)
    if call is None:
        return None, None

    result = env_session.execute_one(call)
    return call, result


def execute_tool_calls(
    *,
    env_session: Any,
    response: Any,
) -> List[Tuple[ToolCall, ToolResult]]:
    """Execute all tool calls from the model response using the given env_session."""
    pairs: List[Tuple[ToolCall, ToolResult]] = []
    for call in extract_tool_calls_from_openai_chat_response(response):
        pairs.append((call, env_session.execute_one(call)))
    return pairs
