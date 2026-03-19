from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

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


def extract_first_tool_call_from_openai_chat_response(response: Any) -> Optional[ToolCall]:
    """
    Extract at most one tool call from OpenAI /chat/completions response.
    """
    tool_calls = getattr(response.choices[0].message, "tool_calls", None)
    if not tool_calls:
        return None

    first = tool_calls[0]
    func = getattr(first, "function", None)
    if not func:
        return None

    name = getattr(func, "name", None)
    if not isinstance(name, str) or not name:
        return None

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

    return ToolCall(name=name, arguments=args_obj, tool_call_id=tool_call_id)


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

