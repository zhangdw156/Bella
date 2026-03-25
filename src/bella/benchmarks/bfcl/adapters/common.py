from __future__ import annotations

from typing import Any, Dict, List, Tuple
import json


def map_function_name(bfcl_name: str) -> str:
    """
    Map BFCL function name to an OpenAI tool name.

    Currently, we apply a minimal rule: replace dots with underscores to
    satisfy OpenAI's name regex and align with BFCL's underscore_to_dot
    behavior for FC models.
    """
    return bfcl_name.replace(".", "_")


def build_tools_from_functions(functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert BFCL function definitions (with JSON schema) into OpenAI tools schema.
    """
    tools: List[Dict[str, Any]] = []
    for func in functions:
        original_name = func.get("name")
        description = func.get("description", "")
        parameters = func.get("parameters", {})

        if not original_name or not parameters:
            continue

        tool_name = map_function_name(original_name)

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return tools


def parse_tool_calls(response: Any) -> List[Dict[str, str]]:
    """
    Parse OpenAI tool_calls into BFCL-compatible result list:
    [{function_name: "<json-arguments>"}].
    """
    tool_calls = getattr(response.choices[0].message, "tool_calls", None)
    if not tool_calls:
        return []

    parsed: List[Dict[str, str]] = []
    for call in tool_calls:
        func = getattr(call, "function", None)
        if not func:
            continue

        name = getattr(func, "name", None)
        arguments = getattr(func, "arguments", "") or ""

        if not name:
            continue

        args_str = arguments
        try:
            args_obj = json.loads(arguments)
            args_str = json.dumps(args_obj, separators=(",", ":"))
        except Exception:
            args_str = arguments

        parsed.append({name: args_str})

    return parsed


def extract_usage(response: Any) -> Tuple[int, int]:
    """
    Extract prompt and completion token counts from OpenAI response. Returns
    (input_tokens, output_tokens).
    """
    usage = getattr(response, "usage", None)
    if not usage:
        return 0, 0
    return getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0)

