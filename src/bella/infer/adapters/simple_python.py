from __future__ import annotations

from typing import Any, Dict, List

from bella.infer.types import BellaRequest, BellaResult
from bella.infer.adapters.base import BFCLAdapter, register_adapter
from bella.infer.adapters.common import (
    build_tools_from_functions,
    parse_tool_calls,
    extract_usage,
)


def _build_messages_and_tools(entry: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    test_id = entry.get("id", "")
    question = entry.get("question", [])
    functions = entry.get("function", [])

    system_content = (
        "You are a function calling model. "
        "Given a user question and a set of available functions, "
        "you must choose the most appropriate function and provide arguments "
        "as a JSON object that satisfies the given JSON schema."
    )

    # Extract first user utterance from BFCL question (single-turn non-live).
    if isinstance(question, list) and question and isinstance(question[0], list):
        first_turn = question[0]
        if first_turn and isinstance(first_turn[0], dict):
            user_text = first_turn[0].get("content", "")
        else:
            user_text = str(question)
    else:
        user_text = str(question)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                f"BFCL test id: {test_id}\n"
                f"User question:\n{user_text}\n\n"
                "Choose ONE function from the provided tools and provide only JSON arguments."
            ),
        },
    ]

    tools = build_tools_from_functions(functions)
    return messages, tools


def _parse_tool_calls(response: Any) -> List[Dict[str, str]]:
    return parse_tool_calls(response)


@register_adapter("simple_python")
class SimplePythonAdapter(BFCLAdapter):
    def build_request(self, entry: Dict[str, Any], state: Dict[str, Any]) -> BellaRequest:
        messages, tools = _build_messages_and_tools(entry)
        return BellaRequest(messages=messages, tools=tools, tool_choice="auto", temperature=0.0)

    def parse_response(
        self,
        entry: Dict[str, Any],
        response: Any,
        state: Dict[str, Any],
    ) -> BellaResult:
        parsed_result = _parse_tool_calls(response)

        input_tokens, output_tokens = extract_usage(response)

        return BellaResult(
            id=entry["id"],
            result=parsed_result,
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            latency=0.0,
        )

    def result_group(self, category: str) -> str:
        return "non_live"

    def result_filename(self, category: str) -> str:
        return "BFCL_v4_simple_python_result.json"

