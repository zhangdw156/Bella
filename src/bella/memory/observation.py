"""
Shared helpers for tool-result memory: truncation and factual observation verbalization.
Used by ToolResultMemoryPlugin and ToolResultMemoryV2Plugin only; not part of the plugin API.
"""
from __future__ import annotations

import json
from typing import Any, Dict


def is_error_output(text: str) -> bool:
    text_l = text.lower()
    return any(
        k in text_l
        for k in [
            "error",
            "failed",
            "no such file",
            "not found",
            "invalid path",
            "exception",
        ]
    )


def truncate_tool_output(
    text: str,
    max_chars_per_item: int,
) -> str:
    """Keep error outputs as complete as possible; otherwise apply hard truncation."""
    if not text:
        return text
    if is_error_output(text):
        hard_limit = max(max_chars_per_item * 2, 1200)
        if len(text) > hard_limit:
            return text[:hard_limit] + "...<truncated>"
        return text
    if len(text) <= max_chars_per_item:
        return text
    return text[:max_chars_per_item] + "...<truncated>"


def _safe_json_obj(text: str) -> Dict[str, Any] | None:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def observation_from_tool_result(tool_call: str, tool_result: str) -> str:
    """
    Convert raw tool result to a factual observation string.
    No planner hints, no extra reasoning.
    """
    call = tool_call or "unknown_tool()"
    tool_name = call.split("(", 1)[0].strip() if "(" in call else call.strip()
    result = tool_result or ""
    obj = _safe_json_obj(result)

    if is_error_output(result):
        if obj and isinstance(obj.get("error"), str):
            return f"{tool_name} failed: {obj['error']}"
        return f"{tool_name} failed: {result}"

    if tool_name == "pwd":
        if obj and isinstance(obj.get("current_working_directory"), str):
            return f"The current working directory is {obj['current_working_directory']}."
        return f"pwd returned: {result}"

    if tool_name == "ls":
        if obj and isinstance(obj.get("current_directory_content"), list):
            items = [str(x) for x in obj["current_directory_content"]]
            if not items:
                return "The current directory is empty."
            return "The current directory contains: " + ", ".join(items) + "."
        return f"ls returned: {result}"

    if tool_name == "cd":
        if obj and isinstance(obj.get("current_working_directory"), str):
            return f"Changed directory to {obj['current_working_directory']}."
        return f"cd returned: {result}"

    if tool_name == "mkdir":
        if result.strip() == "None":
            return "The directory creation command completed."
        if obj and isinstance(obj.get("result"), str):
            return f"mkdir result: {obj['result']}"
        return f"mkdir returned: {result}"

    if tool_name == "mv":
        if obj and isinstance(obj.get("result"), str):
            return f"Move result: {obj['result']}"
        return f"mv returned: {result}"

    if tool_name == "grep":
        if obj and "matches" in obj:
            return "The grep command found matches."
        return f"grep returned: {result}"

    if tool_name == "sort":
        if result.strip() == "None":
            return "The sort command completed."
        return f"sort returned: {result}"

    if tool_name == "diff":
        if obj and isinstance(obj.get("diff_lines"), str):
            if obj["diff_lines"].strip():
                return "The diff command found differences between the files."
            return "The diff command found no differences between the files."
        return f"diff returned: {result}"

    return f"The tool {tool_name} returned: {result}"
