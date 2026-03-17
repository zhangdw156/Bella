from __future__ import annotations

from typing import Any, Dict, List

from bella.infer.types import BellaRequest, BellaResult
from bella.infer.adapters.base import BFCLAdapter, register_adapter
from bella.infer.adapters.common import (
    build_tools_from_functions,
    parse_tool_calls,
    extract_usage,
)


# NOTE: BFCL v4 "multiple" category semantics
#
# Although the category name suggests multiple function calls, the current BFCL v4
# data and `multiple_function_checker` behavior indicate that the correct / safest
# implementation is:
#   - Model output is a list of function calls (model_output)
#   - The evaluator enforces len(model_output) == len(possible_answers)
#   - For BFCL_v4_multiple.json, possible_answers is effectively a list of length 1
#     (one target function per test entry)
#   - `multiple_function_checker` then ONLY inspects model_output[0] against
#     possible_answers[0] via simple_function_checker
#
# Therefore, for now this adapter intentionally returns exactly ONE function call
# in the result list, effectively "choosing the best function among many
# candidates" rather than executing multiple tools. If BFCL later evolves to
# require true multi-call semantics, this adapter will need to be revisited.


def _build_messages_and_tools(entry: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    test_id = entry.get("id", "")
    question = entry.get("question", [])
    functions = entry.get("function", [])

    system_content = (
        "You are a function calling model. "
        "Given a user question and a set of available functions, "
        "choose the most appropriate function and provide arguments "
        "as a JSON object that satisfies the given JSON schema."
    )

    # Extract first user utterance (single-turn non-live).
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
                "Select ONE function from the provided tools and return only its JSON arguments."
            ),
        },
    ]

    tools = build_tools_from_functions(functions)
    return messages, tools


def _parse_tool_calls_single(response: Any) -> List[Dict[str, str]]:
    """
    Parse tool_calls but keep only a single call in the result list, to match
    BFCL v4 multiple_function_checker expectations (len(model_output) == 1).
    """
    all_calls = parse_tool_calls(response)
    if not all_calls:
        return []
    # Only keep the first call to satisfy multiple_function_checker semantics.
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
        # multiple 属于非 live AST 类型
        return "non_live"

    def result_filename(self, category: str) -> str:
        return "BFCL_v4_multiple_result.json"

