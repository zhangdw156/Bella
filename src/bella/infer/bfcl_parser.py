from __future__ import annotations

from typing import Any, Dict, List
import json


def parse_simple_python_tool_calls(response: Any) -> List[Dict[str, str]]:
    """
    Parse tool_calls from an OpenAI chat completion response into the
    BFCL-compatible `result` structure for simple_python.

    BFCL simple_python expects a list of dicts, where each dict maps
    function name -> JSON-stringified arguments, e.g.:
      [{"calculate_triangle_area": "{\"base\": 10, \"height\": 5}"}]
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

        # arguments 应该已经是 JSON 字符串，但为防万一尝试标准化一下。
        args_str = arguments
        try:
            args_obj = json.loads(arguments)
            # 重新 dump 一遍，保证是紧凑 JSON。
            args_str = json.dumps(args_obj, separators=(",", ":"))
        except Exception:
            # 如果不是合法 JSON，就保留原始字符串，让 evaluator 自行处理错误。
            args_str = arguments

        parsed.append({name: args_str})

    return parsed

