from __future__ import annotations

from typing import Any, Dict, List
import os

from bella.infer.types import BellaRequest, BellaResult
from bella.infer.adapters.base import BFCLAdapter, register_adapter
from bella.infer.adapters.common import build_tools_from_functions, extract_usage


def _extract_turn_user_texts(question: Any) -> List[str]:
    """
    BFCL multi_turn_base `question` is a list of turns, each turn is a list of
    messages (role/content). For now we keep the prompt simple and only pass
    the user content of the current turn into the LLM, while preserving a
    short system instruction.
    """
    turn_texts: List[str] = []
    if not isinstance(question, list):
        return [str(question)]

    for turn in question:
        if isinstance(turn, list) and turn:
            # Find first user message in this turn
            user_msg = None
            for msg in turn:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_msg = msg
                    break
            if user_msg is not None:
                turn_texts.append(str(user_msg.get("content", "")))
            else:
                turn_texts.append(str(turn))
        else:
            turn_texts.append(str(turn))
    return turn_texts


@register_adapter("multi_turn_base")
class MultiTurnBaseAdapter(BFCLAdapter):
    """
    Adapter for BFCL v4 `multi_turn_base` category.

    State layout:
      state = {
          "conversation": {
              "current_turn_index": int,
              "turn_texts": list[str],
              "messages": list[dict],        # messages for current turn
              "model_responses": list[list[Any]],  # optional raw responses per turn
              "history_calls": list[list[str]],    # human-readable per-turn calls
          },
          "execution": {
              # per-turn list of raw FC-style function calls, where each item is
              # a dict {name: "<json-args>"} compatible with BFCL convert_to_function_call.
              "function_calls": list[list[dict[str, str]]],
          },
      }
    """

    def __init__(self) -> None:
        # Minimal memory strategy switch:
        #   - "none": baseline, no history injected
        #   - "action_history": inject previous turns' function calls into prompt
        self.memory_mode: str = os.getenv("BELLA_MULTI_TURN_MEMORY_MODE", "none")

    def init_state(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        question = entry.get("question", [])
        ground_truth = entry.get("ground_truth", [])

        turn_texts = _extract_turn_user_texts(question)
        num_turns = max(len(turn_texts), len(ground_truth))

        # conversation state
        conversation: Dict[str, Any] = {
            "current_turn_index": 0,
            "turn_texts": turn_texts,
            "messages": [],
            "model_responses": [[] for _ in range(num_turns)],
            "history_calls": [[] for _ in range(num_turns)],
        }

        # execution state: per-turn list of raw FC-style function calls,
        # where each item is a dict {name: "<json-args>"} compatible with
        # BFCL convert_to_function_call for FC models.
        execution: Dict[str, Any] = {
            "function_calls": [[] for _ in range(num_turns)],
        }

        return {
            "conversation": conversation,
            "execution": execution,
        }

    def _build_messages_for_turn(self, entry: Dict[str, Any], state: Dict[str, Any]) -> List[Dict[str, Any]]:
        conversation = state["conversation"]
        test_id = entry.get("id", "")
        turn_index: int = conversation["current_turn_index"]
        turn_texts: List[str] = conversation["turn_texts"]

        # Bound check
        if turn_index < len(turn_texts):
            user_text = turn_texts[turn_index]
        else:
            user_text = ""

        system_content = (
            "You are a function calling model operating in multiple turns. "
            "At each turn, you must decide which file-system function to call "
            "and provide arguments as a JSON object that satisfies the given "
            "JSON schema. Only output a single function call via tools."
        )

        # Baseline user content
        user_content = (
            f"BFCL test id: {test_id}\n"
            f"Current turn index: {turn_index}\n"
            f"User request for this turn:\n{user_text}\n"
        )

        # Optionally inject action history (previous turns only)
        if self.memory_mode == "action_history" and turn_index > 0:
            history_calls: List[List[str]] = conversation.get("history_calls", [])
            history_lines: List[str] = []
            for idx in range(min(turn_index, len(history_calls))):
                if not history_calls[idx]:
                    continue
                joined = "; ".join(history_calls[idx])
                history_lines.append(f"- Turn {idx}: {joined}")
            if history_lines:
                user_content += "\nAction history so far:\n" + "\n".join(history_lines) + "\n"

        user_content += (
            "\nUse exactly one tool function call at this turn and "
            "only provide JSON arguments."
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": user_content,
            },
        ]

        conversation["messages"] = messages
        return messages

    def build_request(self, entry: Dict[str, Any], state: Dict[str, Any]) -> BellaRequest:
        conversation = state["conversation"]
        execution = state["execution"]

        # Determine total number of turns from execution.function_calls length
        num_turns = len(execution["function_calls"])
        current_turn_index = conversation["current_turn_index"]
        if current_turn_index >= num_turns:
            # Defensive: should not be called if no more turns
            current_turn_index = num_turns - 1
            conversation["current_turn_index"] = current_turn_index

        messages = self._build_messages_for_turn(entry, state)

        # BFCL utils.populate_test_cases_with_predefined_functions has already
        # injected function docs into entry["function"].
        functions = entry.get("function", [])
        tools = build_tools_from_functions(functions)

        return BellaRequest(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )

    def _append_function_call_for_turn(
        self,
        entry: Dict[str, Any],
        response: Any,
        state: Dict[str, Any],
    ) -> None:
        """
        Extract exactly one tool call from the current response and append
        its canonical string form to execution.function_calls[current_turn_index].

        We purposely use a single function call per step (per response),
        and encode it as one string, e.g. "cd(folder='document')".
        """
        conversation = state["conversation"]
        execution = state["execution"]

        turn_index: int = conversation["current_turn_index"]
        function_calls: List[List[Dict[str, str]]] = execution["function_calls"]

        # Track raw response for potential debugging / future extensions
        conversation["model_responses"][turn_index].append(response)

        # OpenAI responses with tools expose choices[0].message.tool_calls
        tool_calls = getattr(response.choices[0].message, "tool_calls", None)
        if not tool_calls:
            return

        # We only care about the first tool call at this turn (one call per step).
        first_call = tool_calls[0]
        func = getattr(first_call, "function", None)
        if not func:
            return

        name = getattr(func, "name", None)
        arguments = getattr(func, "arguments", "") or ""
        if not name:
            return

        # arguments is a JSON string; we want to store FC-compatible raw
        # function calls to match BFCL's convert_to_function_call input:
        # list[dict[str, str]] where the value is a JSON string.
        try:
            import json

            args_obj = json.loads(arguments)
        except Exception:
            # If parsing fails, store the raw argument string as-is.
            function_calls[turn_index].append({name: arguments})
            return

        if not isinstance(args_obj, dict):
            function_calls[turn_index].append({name: json.dumps(args_obj, separators=(",", ":"))})
            return

        # Store as JSON string to mirror parse_tool_calls & simple_python style.
        args_json = json.dumps(args_obj, separators=(",", ":"))
        function_calls[turn_index].append({name: args_json})

    def _format_execution_history(self, execution: Dict[str, Any]) -> List[List[str]]:
        """
        Convert execution.function_calls (FC raw dicts) into human-readable
        strings, e.g. {"pwd": "{}"} -> ["pwd()"], for prompt injection.
        This is the single source of truth to avoid drift between evaluator
        raw schema and prompt-side representation.
        """
        import json

        formatted: List[List[str]] = []
        raw_calls: List[List[Dict[str, str]]] = execution.get("function_calls", [])

        for turn_calls in raw_calls:
            turn_strings: List[str] = []
            for call in turn_calls:
                if not isinstance(call, dict):
                    continue
                for name, args_json in call.items():
                    try:
                        args_obj = json.loads(args_json)
                    except Exception:
                        # Fallback: keep raw json as a single argument
                        turn_strings.append(f"{name}({args_json!r})")
                        continue

                    if not isinstance(args_obj, dict) or not args_obj:
                        turn_strings.append(f"{name}()")
                        continue

                    parts: List[str] = []
                    for k, v in args_obj.items():
                        parts.append(f"{k}={repr(v)}")
                    turn_strings.append(f"{name}({', '.join(parts)})")
            formatted.append(turn_strings)
        return formatted

    def parse_response(
        self,
        entry: Dict[str, Any],
        response: Any,
        state: Dict[str, Any],
    ) -> BellaResult:
        self._append_function_call_for_turn(entry, response, state)

        input_tokens, output_tokens = extract_usage(response)

        execution = state["execution"]
        function_calls: List[List[Dict[str, str]]] = execution["function_calls"]

        # Refresh human-readable history from the single raw source of truth.
        conversation = state["conversation"]
        conversation["history_calls"] = self._format_execution_history(execution)

        return BellaResult(
            id=entry["id"],
            result=function_calls,
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            latency=0.0,
        )

    def has_next_turn(self, entry: Dict[str, Any], state: Dict[str, Any]) -> bool:
        conversation = state["conversation"]
        execution = state["execution"]

        current_turn_index: int = conversation["current_turn_index"]
        num_turns = len(execution["function_calls"])

        # Stop when we've just finished the last turn.
        if current_turn_index >= num_turns - 1:
            return False

        # Advance to next turn for the next build_request call.
        conversation["current_turn_index"] = current_turn_index + 1
        return True

    def finalize_result(
        self,
        entry: Dict[str, Any],
        state: Dict[str, Any],
        last_result: BellaResult,
    ) -> BellaResult:
        # Ensure BellaResult.result is exactly the accumulated function_calls.
        execution = state["execution"]
        function_calls: List[List[str]] = execution["function_calls"]

        return BellaResult(
            id=entry["id"],
            result=function_calls,
            input_token_count=last_result.input_token_count,
            output_token_count=last_result.output_token_count,
            latency=last_result.latency,
            extra=last_result.extra,
        )

    def result_group(self, category: str) -> str:
        # multi_turn_base 属于 multi_turn 分组
        return "multi_turn"

    def result_filename(self, category: str) -> str:
        return "BFCL_v4_multi_turn_base_result.json"

