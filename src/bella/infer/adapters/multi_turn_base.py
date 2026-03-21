from __future__ import annotations

from typing import Any, Dict, List
import os
import json

from bella.bfcl_resources import load_bfcl_categories, load_prompt_system, render_user_prompt
from bella.env.bfcl_multi_turn import BFCLMultiTurnEnvironmentSession
from bella.env.tool_executor import execute_first_tool_call
from bella.memory import get_plugin
from bella.infer.types import BellaRequest, BellaResult
from bella.infer.adapters.base import BFCLAdapter, register_adapter
from bella.infer.adapters.common import build_tools_from_functions, extract_usage

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
        }
        self.memory_plugin.init_state(conversation)

        execution: Dict[str, Any] = {
            "function_calls": [[] for _ in range(num_turns)],
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

        blocks = self.memory_plugin.build_prompt_blocks(entry, state, turn_index)
        user_content = render_user_prompt(
            "multi_turn_base",
            test_id=test_id,
            turn_index=turn_index,
            user_text=user_text,
            **blocks,
        )

        history_messages: List[Dict[str, Any]] = conversation["history_messages"]
        if not history_messages:
            system_content = load_prompt_system("multi_turn_base")
            history_messages.append({"role": "system", "content": system_content})

        history_messages.append({"role": "user", "content": user_content})

    def build_request(self, entry: Dict[str, Any], state: Dict[str, Any]) -> BellaRequest:
        conversation = state["conversation"]
        execution = state["execution"]

        num_turns = len(execution["function_calls"])
        current_turn_index = conversation["current_turn_index"]
        if current_turn_index >= num_turns:
            current_turn_index = num_turns - 1
            conversation["current_turn_index"] = current_turn_index

        self._append_user_message_for_turn(entry, state)
        messages: List[Dict[str, Any]] = conversation["history_messages"]

        functions = entry.get("function", [])
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

    def _append_function_call_for_turn(
        self,
        entry: Dict[str, Any],
        response: Any,
        state: Dict[str, Any],
    ) -> None:
        """Extract exactly one tool call from the current response and append to execution.function_calls."""
        conversation = state["conversation"]
        execution = state["execution"]

        turn_index: int = conversation["current_turn_index"]
        function_calls: List[List[Dict[str, str]]] = execution["function_calls"]

        conversation["model_responses"][turn_index].append(response)

        tool_calls = getattr(response.choices[0].message, "tool_calls", None)
        if not tool_calls:
            return

        first_call = tool_calls[0]
        func = getattr(first_call, "function", None)
        if not func:
            return

        name = getattr(func, "name", None)
        arguments = getattr(func, "arguments", "") or ""
        if not name:
            return

        try:
            args_obj = json.loads(arguments)
        except Exception:
            function_calls[turn_index].append({name: arguments})
            return

        if not isinstance(args_obj, dict):
            function_calls[turn_index].append({name: json.dumps(args_obj, separators=(",", ":"))})
            return

        args_json = json.dumps(args_obj, separators=(",", ":"))
        function_calls[turn_index].append({name: args_json})

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
        self._append_function_call_for_turn(entry, response, state)

        input_tokens, output_tokens = extract_usage(response)

        execution = state["execution"]
        function_calls: List[List[Dict[str, str]]] = execution["function_calls"]

        conversation = state["conversation"]
        conversation["history_calls"] = self._format_execution_history(execution)

        env_session = state.get("env")
        tool_call, tool_result = execute_first_tool_call(env_session=env_session, response=response)
        if tool_result is not None:
            assistant_content = getattr(response.choices[0].message, "content", "") or ""
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": assistant_content}
            if tool_call is not None:
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
                ]
            conversation["history_messages"].append(assistant_msg)

            conversation["history_messages"].append(
                {
                    "role": "tool",
                    "content": tool_result.output,
                    **({"tool_call_id": tool_result.tool_call_id} if tool_result.tool_call_id else {}),
                }
            )
            turn_index = conversation["current_turn_index"]
            if 0 <= turn_index < len(conversation.get("tool_outputs", [])):
                conversation["tool_outputs"][turn_index] = tool_result.output

            selected_calls = conversation.get("history_calls", [])
            tool_call_str = ""
            if 0 <= turn_index < len(selected_calls) and selected_calls[turn_index]:
                tool_call_str = str(selected_calls[turn_index][0])
            elif tool_call is not None:
                tool_call_str = f"{tool_call.name}(...)"
            self.memory_plugin.on_tool_result(
                conversation,
                turn_index,
                tool_call_str,
                str(tool_result.output),
            )

        if self.debug_mode:
            turn_index = conversation["current_turn_index"]
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
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            latency=0.0,
        )

    def has_next_turn(self, entry: Dict[str, Any], state: Dict[str, Any]) -> bool:
        conversation = state["conversation"]
        execution = state["execution"]

        current_turn_index = conversation["current_turn_index"]
        num_turns = len(execution["function_calls"])

        if current_turn_index >= num_turns - 1:
            return False

        conversation["current_turn_index"] = current_turn_index + 1
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
            input_token_count=last_result.input_token_count,
            output_token_count=last_result.output_token_count,
            latency=last_result.latency,
            extra=last_result.extra,
        )

    def result_group(self, category: str) -> str:
        return _CATEGORIES.get(category, {}).get("result_group", "multi_turn")

    def result_filename(self, category: str) -> str:
        return _CATEGORIES.get(category, {}).get("result_filename", "BFCL_v4_multi_turn_base_result.json")
