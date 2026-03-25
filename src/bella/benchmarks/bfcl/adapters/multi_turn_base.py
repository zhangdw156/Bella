from __future__ import annotations

from typing import Any, Dict, List
import os
import json

from bella.benchmarks.bfcl.resources import load_bfcl_categories, load_prompt_system, render_user_prompt
from bella.benchmarks.bfcl.env.multi_turn import BFCLMultiTurnEnvironmentSession
from bella.benchmarks.bfcl.env.tool_executor import execute_tool_calls
from bella.memory import get_plugin
from bella.infer.types import BellaRequest, BellaResult
from bella.benchmarks.bfcl.adapters.base import BFCLAdapter, register_adapter
from bella.benchmarks.bfcl.adapters.common import build_tools_from_functions, extract_usage
from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC

_CATEGORIES = load_bfcl_categories()


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


@register_adapter("multi_turn_long_context")
@register_adapter("multi_turn_miss_param")
@register_adapter("multi_turn_miss_func")
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
              "history_calls": list[list[str]],    # human-readable per-turn calls (synced from execution)
          },
          "execution": {
              # per-turn list of raw FC-style function calls, where each item is
              # a dict {name: "<json-args>"} compatible with BFCL convert_to_function_call.
              "function_calls": list[list[dict[str, str]]],
          },
      }
    """

    def __init__(self) -> None:
        self.memory_plugin = get_plugin()
        self._memory_mode: str = os.getenv("BELLA_MULTI_TURN_MEMORY_MODE", "none").strip().lower()
        self.max_steps_per_turn: int = int(os.getenv("BELLA_MULTI_TURN_MAX_STEPS_PER_TURN", "8"))
        self.debug_mode: bool = os.getenv("BELLA_MULTI_TURN_DEBUG", "0") not in (
            "0",
            "",
            "false",
            "False",
        )
        self.env_model_name: str = os.getenv("BELLA_MULTI_TURN_ENV_MODEL_NAME", "bella")

    def init_state(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        question = entry.get("question", [])
        ground_truth = entry.get("ground_truth", [])

        turn_texts = _extract_turn_user_texts(question)
        num_turns = max(len(turn_texts), len(ground_truth))

        conversation: Dict[str, Any] = {
            "current_turn_index": 0,
            "turn_texts": turn_texts,
            "history_messages": [],
            "model_responses": [[] for _ in range(num_turns)],
            "history_calls": [[] for _ in range(num_turns)],
            "tools_per_turn": [[] for _ in range(num_turns)],
            "tool_outputs": ["" for _ in range(num_turns)],
            "turn_user_appended": [False for _ in range(num_turns)],
            "turn_step_counts": [0 for _ in range(num_turns)],
            "turn_complete": [False for _ in range(num_turns)],
            "finished": False,
        }
        self.memory_plugin.init_state(conversation)

        execution: Dict[str, Any] = {
            "function_calls": [[] for _ in range(num_turns)],
            "functions": list(entry.get("function", [])),
            "holdout_injected_turns": set(),
        }

        usage: Dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
        }

        env_session = BFCLMultiTurnEnvironmentSession(
            initial_config=entry.get("initial_config", {}) or {},
            involved_classes=entry.get("involved_classes", []) or [],
            model_name=self.env_model_name,
            test_entry_id=str(entry.get("id", "")),
            long_context=("long_context" in str(entry.get("id", "")) or "composite" in str(entry.get("id", ""))),
        )

        return {
            "conversation": conversation,
            "execution": execution,
            "env": env_session,
            "usage": usage,
        }

    def _append_user_message_for_turn(self, entry: Dict[str, Any], state: Dict[str, Any]) -> None:
        conversation = state["conversation"]
        test_id = entry.get("id", "")
        turn_index: int = conversation["current_turn_index"]
        turn_texts: List[str] = conversation["turn_texts"]

        if turn_index < len(turn_texts):
            user_text = turn_texts[turn_index]
        else:
            user_text = ""

        missed_function = entry.get("missed_function", {})
        if not user_text and str(turn_index) in missed_function:
            user_text = DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC

        blocks = self.memory_plugin.build_prompt_blocks(entry, state, turn_index)
        user_content = render_user_prompt(
            "multi_turn",
            test_id=test_id,
            turn_index=turn_index,
            user_text=user_text,
            **blocks,
        )

        history_messages: List[Dict[str, Any]] = conversation["history_messages"]
        if not history_messages:
            system_content = load_prompt_system("multi_turn")
            history_messages.append({"role": "system", "content": system_content})

        history_messages.append({"role": "user", "content": user_content})
        conversation["turn_user_appended"][turn_index] = True

    def build_request(self, entry: Dict[str, Any], state: Dict[str, Any]) -> BellaRequest:
        conversation = state["conversation"]
        execution = state["execution"]

        num_turns = len(execution["function_calls"])
        current_turn_index = conversation["current_turn_index"]
        if current_turn_index >= num_turns:
            current_turn_index = num_turns - 1
            conversation["current_turn_index"] = current_turn_index

        missed_function = entry.get("missed_function", {})
        holdout_turns = execution.get("holdout_injected_turns", set())
        holdout_turn_key = str(current_turn_index)
        if holdout_turn_key in missed_function and holdout_turn_key not in holdout_turns:
            current_functions = execution.get("functions", [])
            for fn_name in missed_function[holdout_turn_key]:
                if any(fn.get("name") == fn_name for fn in current_functions):
                    continue
                for fn in entry.get("function", []):
                    if fn.get("name") == fn_name:
                        current_functions.append(fn)
                        break
            execution["functions"] = current_functions
            holdout_turns.add(holdout_turn_key)

        if not conversation["turn_user_appended"][current_turn_index]:
            self._append_user_message_for_turn(entry, state)
        messages: List[Dict[str, Any]] = conversation["history_messages"]

        functions = execution.get("functions", entry.get("function", []))
        tools = build_tools_from_functions(functions)

        tool_names: List[str] = []
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name")
            if isinstance(name, str):
                tool_names.append(name)
        tools_per_turn: List[List[str]] = conversation.get("tools_per_turn", [])
        if not tools_per_turn or len(tools_per_turn) != len(execution["function_calls"]):
            tools_per_turn = [[] for _ in range(len(execution["function_calls"]))]
            conversation["tools_per_turn"] = tools_per_turn
        tools_per_turn[current_turn_index] = tool_names

        return BellaRequest(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )

    def _append_function_calls_for_turn(
        self,
        entry: Dict[str, Any],
        response: Any,
        state: Dict[str, Any],
    ) -> int:
        """Extract all tool calls from the current response and append them to execution.function_calls."""
        conversation = state["conversation"]
        execution = state["execution"]

        turn_index: int = conversation["current_turn_index"]
        function_calls: List[List[Dict[str, str]]] = execution["function_calls"]

        conversation["model_responses"][turn_index].append(response)

        tool_calls = getattr(response.choices[0].message, "tool_calls", None)
        if not tool_calls:
            return 0

        appended = 0
        for call in tool_calls:
            func = getattr(call, "function", None)
            if not func:
                continue

            name = getattr(func, "name", None)
            arguments = getattr(func, "arguments", "") or ""
            if not name:
                continue

            try:
                args_obj = json.loads(arguments)
            except Exception:
                function_calls[turn_index].append({name: arguments})
                appended += 1
                continue

            if not isinstance(args_obj, dict):
                function_calls[turn_index].append({name: json.dumps(args_obj, separators=(",", ":"))})
                appended += 1
                continue

            args_json = json.dumps(args_obj, separators=(",", ":"))
            function_calls[turn_index].append({name: args_json})
            appended += 1

        return appended

    def _format_execution_history(self, execution: Dict[str, Any]) -> List[List[str]]:
        """Convert execution.function_calls into human-readable strings for conversation.history_calls."""
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
        appended_count = self._append_function_calls_for_turn(entry, response, state)

        input_tokens, output_tokens = extract_usage(response)
        state["usage"]["input_tokens"] += input_tokens
        state["usage"]["output_tokens"] += output_tokens

        execution = state["execution"]
        function_calls: List[List[Dict[str, str]]] = execution["function_calls"]

        conversation = state["conversation"]
        conversation["history_calls"] = self._format_execution_history(execution)
        turn_index = conversation["current_turn_index"]
        conversation["turn_step_counts"][turn_index] += 1

        env_session = state.get("env")
        executed_pairs = execute_tool_calls(env_session=env_session, response=response)
        assistant_content = getattr(response.choices[0].message, "content", "") or ""
        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": assistant_content}
        if executed_pairs:
            assistant_msg["tool_calls"] = [
                {
                    "id": tool_call.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(
                            tool_call.arguments, separators=(",", ":"), ensure_ascii=False
                        ),
                    },
                }
                for tool_call, _ in executed_pairs
            ]
        conversation["history_messages"].append(assistant_msg)

        if executed_pairs:
            last_tool_output = ""
            selected_calls = conversation.get("history_calls", [])
            selected_for_turn = (
                selected_calls[turn_index] if 0 <= turn_index < len(selected_calls) else []
            )
            for pair_index, (tool_call, tool_result) in enumerate(executed_pairs):
                conversation["history_messages"].append(
                    {
                        "role": "tool",
                        "content": tool_result.output,
                        **({"tool_call_id": tool_result.tool_call_id} if tool_result.tool_call_id else {}),
                    }
                )
                tool_call_str = (
                    str(selected_for_turn[pair_index])
                    if pair_index < len(selected_for_turn)
                    else f"{tool_call.name}(...)"
                )
                self.memory_plugin.on_tool_result(
                    conversation,
                    turn_index,
                    tool_call_str,
                    str(tool_result.output),
                )
                last_tool_output = tool_result.output

            if 0 <= turn_index < len(conversation.get("tool_outputs", [])):
                conversation["tool_outputs"][turn_index] = last_tool_output

        turn_complete = appended_count == 0 or not executed_pairs
        too_many_steps = conversation["turn_step_counts"][turn_index] >= self.max_steps_per_turn
        if too_many_steps:
            turn_complete = True
        conversation["turn_complete"][turn_index] = turn_complete

        if self.debug_mode:
            turn_texts = conversation.get("turn_texts", [])
            user_text = turn_texts[turn_index] if turn_index < len(turn_texts) else ""
            history_calls = conversation.get("history_calls", [])
            tools_per_turn = conversation.get("tools_per_turn", [])
            tool_outputs = conversation.get("tool_outputs", [])
            available_tools = tools_per_turn[turn_index] if turn_index < len(tools_per_turn) else []
            selected_calls = history_calls[turn_index] if turn_index < len(history_calls) else []
            tool_output = tool_outputs[turn_index] if turn_index < len(tool_outputs) else ""

            print(f"[Bella][multi_turn_base][debug] turn={turn_index}")
            print(f"[Bella][multi_turn_base][debug] user_request={user_text!r}")
            print(f"[Bella][multi_turn_base][debug] tools={available_tools}")
            # Legacy debug format (pre memory plugin): action_history uses repr(list of "Turn i: ...").
            if self._memory_mode == "action_history" and turn_index > 0:
                prev_history_lines: List[str] = []
                for idx in range(turn_index):
                    if idx < len(history_calls) and history_calls[idx]:
                        prev_history_lines.append(
                            f"Turn {idx}: {', '.join(history_calls[idx])}"
                        )
                print(
                    "[Bella][multi_turn_base][debug] injected_action_history="
                    + (prev_history_lines or ["<none>"]).__repr__()
                )
            else:
                print("[Bella][multi_turn_base][debug] injected_action_history=<none>")
            mode = self._memory_mode
            if mode in ("tool_result_memory", "tool_result_memory_v2"):
                inner_fn = getattr(self.memory_plugin, "debug_tool_memory_inner", None)
                block = inner_fn(state, turn_index) if callable(inner_fn) else ""
                print(
                    f"[Bella][multi_turn_base][debug] injected_{mode}="
                    + (block if block else "<none>")
                )
            else:
                print("[Bella][multi_turn_base][debug] injected_tool_result_memory=<none>")
            print(f"[Bella][multi_turn_base][debug] selected_calls={selected_calls}")
            if selected_calls:
                print(f"[Bella][multi_turn_base][debug] tool_call_first={selected_calls[0]}")
            print(f"[Bella][multi_turn_base][debug] tool_response={tool_output!r}")
            tail = conversation["history_messages"][-3:]
            print(f"[Bella][multi_turn_base][debug] next_messages_tail={tail!r}")

        return BellaResult(
            id=entry["id"],
            result=function_calls,
            input_token_count=state["usage"]["input_tokens"],
            output_token_count=state["usage"]["output_tokens"],
            latency=0.0,
        )

    def has_next_turn(self, entry: Dict[str, Any], state: Dict[str, Any]) -> bool:
        conversation = state["conversation"]
        execution = state["execution"]

        current_turn_index = conversation["current_turn_index"]
        num_turns = len(execution["function_calls"])

        if not conversation["turn_complete"][current_turn_index]:
            return True

        if current_turn_index >= num_turns - 1:
            conversation["finished"] = True
            return False

        conversation["current_turn_index"] = current_turn_index + 1
        next_turn_index = conversation["current_turn_index"]
        conversation["turn_user_appended"][next_turn_index] = False
        return True

    def finalize_result(
        self,
        entry: Dict[str, Any],
        state: Dict[str, Any],
        last_result: BellaResult,
    ) -> BellaResult:
        execution = state["execution"]
        function_calls = execution["function_calls"]

        return BellaResult(
            id=entry["id"],
            result=function_calls,
            input_token_count=state["usage"]["input_tokens"],
            output_token_count=state["usage"]["output_tokens"],
            latency=last_result.latency,
            extra=last_result.extra,
        )

    def result_group(self, category: str) -> str:
        return _CATEGORIES.get(category, {}).get("result_group", "multi_turn")

    def result_filename(self, category: str) -> str:
        return _CATEGORIES.get(category, {}).get("result_filename", "BFCL_v4_multi_turn_base_result.json")
