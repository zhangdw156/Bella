from __future__ import annotations

from typing import Any, Dict, List

from bella.benchmarks.bfcl.resources import load_bfcl_categories, load_prompt_system, render_user_prompt
from bella.infer.types import BellaRequest, BellaResult
from bella.benchmarks.bfcl.adapters.base import BFCLAdapter, register_adapter
from bella.benchmarks.bfcl.adapters.common import (
    build_tools_from_functions,
    parse_tool_calls,
    extract_usage,
)

# NOTE: BFCL v4 "multiple" category semantics
#
# Although the category name suggests multiple function calls, the current BFCL v4
# data and `multiple_function_checker` behavior indicate that the correct / safest
# implementation is to return exactly ONE function call.  See original module
# docstring for details.

_CATEGORIES = load_bfcl_categories()


def _extract_user_text(question: Any) -> str:
    """Extract first user utterance (single-turn non-live)."""
    if isinstance(question, list) and question and isinstance(question[0], list):
        first_turn = question[0]
        if first_turn and isinstance(first_turn[0], dict):
            return first_turn[0].get("content", "")
        return str(question)
    return str(question)


def _build_messages_and_tools(entry: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    test_id = entry.get("id", "")
    question = entry.get("question", [])
    functions = entry.get("function", [])

    system_content = load_prompt_system("multiple")
    user_text = _extract_user_text(question)
    user_content = render_user_prompt("multiple", test_id=test_id, user_text=user_text)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    tools = build_tools_from_functions(functions)
    return messages, tools


def _parse_tool_calls_single(response: Any) -> List[Dict[str, str]]:
    all_calls = parse_tool_calls(response)
    if not all_calls:
        return []
    return [all_calls[0]]


@register_adapter("multiple")
class MultipleAdapter(BFCLAdapter):
    def build_request(self, entry: Dict[str, Any], state: Dict[str, Any]) -> BellaRequest:
        messages, tools = _build_messages_and_tools(entry)
        return BellaRequest(messages=messages, tools=tools, tool_choice="auto", temperature=0.0)

    def parse_response(
        self,
        entry: Dict[str, Any],
        response: Any,
        state: Dict[str, Any],
    ) -> BellaResult:
        parsed_result = _parse_tool_calls_single(response)
        input_tokens, output_tokens = extract_usage(response)

        return BellaResult(
            id=entry["id"],
            result=parsed_result,
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            latency=0.0,
        )

    def result_group(self, category: str) -> str:
        return _CATEGORIES.get(category, {}).get("result_group", "non_live")

    def result_filename(self, category: str) -> str:
        return _CATEGORIES.get(category, {}).get("result_filename", "BFCL_v4_multiple_result.json")
