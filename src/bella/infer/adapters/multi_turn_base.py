from __future__ import annotations

from typing import Any, Dict, List
import os
import json

from bella.bfcl_resources import load_bfcl_categories, load_prompt_system, render_user_prompt
from bella.env.bfcl_multi_turn import BFCLMultiTurnEnvironmentSession
from bella.env.tool_executor import execute_first_tool_call
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
        # Lightweight per-turn debug trace switch (stdout only).
        self.debug_mode: bool = os.getenv("BELLA_MULTI_TURN_DEBUG", "0") not in (
            "0",
            "",
            "false",
            "False",
        )
        # A stable identifier used by BFCL env executor to cache instances.
        # Keep it independent from the OpenAI model name to avoid accidental mixing.
        self.env_model_name: str = os.getenv("BELLA_MULTI_TURN_ENV_MODEL_NAME", "bella")
        self.tool_result_max_chars_per_item: int = int(
            os.getenv("BELLA_TOOL_RESULT_MEMORY_MAX_CHARS_PER_ITEM", "600")
        )
        self.tool_result_max_items: int = int(
            os.getenv("BELLA_TOOL_RESULT_MEMORY_MAX_ITEMS", "3")
        )
        self.tool_result_max_total_chars: int = int(
            os.getenv("BELLA_TOOL_RESULT_MEMORY_MAX_TOTAL_CHARS", "1800")
        )

    def init_state(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        question = entry.get("question", [])
        ground_truth = entry.get("ground_truth", [])

        turn_texts = _extract_turn_user_texts(question)
        num_turns = max(len(turn_texts), len(ground_truth))

        # conversation state
        conversation: Dict[str, Any] = {
            "current_turn_index": 0,
            "turn_texts": turn_texts,
            # Full multi-turn message history (system/user/tool/assistant).
            "history_messages": [],
            "model_responses": [[] for _ in range(num_turns)],
            "history_calls": [[] for _ in range(num_turns)],
            "tools_per_turn": [[] for _ in range(num_turns)],
            "tool_outputs": ["" for _ in range(num_turns)],
            # memory entries: {"turn": int, "tool_call": str, "tool_result": str}
            "tool_result_memory_items": [],
        }

        # execution state: per-turn list of raw FC-style function calls,
        # where each item is a dict {name: "<json-args>"} compatible with
        # BFCL convert_to_function_call for FC models.
        execution: Dict[str, Any] = {
            "function_calls": [[] for _ in range(num_turns)],
        }

        # Reuse BFCL backend implementations and semantics for environment execution.
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

        # Bound check
        if turn_index < len(turn_texts):
            user_text = turn_texts[turn_index]
        else:
            user_text = ""

        # Optional action history block (previous turns only)
        action_history_section = ""
        tool_result_memory_section = ""
        if self.memory_mode == "action_history" and turn_index > 0:
            history_calls: List[List[str]] = conversation.get("history_calls", [])
            history_lines: List[str] = []
            for idx in range(min(turn_index, len(history_calls))):
                if not history_calls[idx]:
                    continue
                joined = "; ".join(history_calls[idx])
                history_lines.append(f"- Turn {idx}: {joined}")
            if history_lines:
                action_history_section = "\nAction history so far:\n" + "\n".join(history_lines) + "\n"
        elif self.memory_mode == "tool_result_memory" and turn_index > 0:
            memory_block = self._build_tool_result_memory_block(state, turn_index)
            if memory_block:
                tool_result_memory_section = (
                    "\nTool result memory so far (focus on results):\n"
                    + memory_block
                    + "\n"
                )
        elif self.memory_mode == "tool_result_memory_v2" and turn_index > 0:
            memory_block = self._build_tool_result_memory_block(state, turn_index)
            if memory_block:
                tool_result_memory_section = (
                    "\nTool result observations so far:\n"
                    + memory_block
                    + "\n"
                )

        user_content = render_user_prompt(
            "multi_turn_base",
            test_id=test_id,
            turn_index=turn_index,
            user_text=user_text,
            action_history_section=action_history_section,
            tool_result_memory_section=tool_result_memory_section,
        )

        history_messages: List[Dict[str, Any]] = conversation["history_messages"]
        if not history_messages:
            system_content = load_prompt_system("multi_turn_base")
            history_messages.append({"role": "system", "content": system_content})

        history_messages.append({"role": "user", "content": user_content})

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

        # Append current turn user message to the running history (system once).
        self._append_user_message_for_turn(entry, state)
        messages: List[Dict[str, Any]] = conversation["history_messages"]

        # BFCL utils.populate_test_cases_with_predefined_functions has already
        # injected function docs into entry["function"].
        functions = entry.get("function", [])
        tools = build_tools_from_functions(functions)

        # Cache available tools (by name) for this turn for debug trace.
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

    def _is_error_output(self, text: str) -> bool:
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

    def _truncate_tool_output(self, text: str) -> str:
        """
        Keep error outputs as complete as possible; otherwise apply hard truncation.
        """
        if not text:
            return text
        if self._is_error_output(text):
            # Error feedback is high value; keep full text unless extremely long.
            hard_limit = max(self.tool_result_max_chars_per_item * 2, 1200)
            if len(text) > hard_limit:
                return text[:hard_limit] + "...<truncated>"
            return text

        if len(text) <= self.tool_result_max_chars_per_item:
            return text
        return text[: self.tool_result_max_chars_per_item] + "...<truncated>"

    def _build_tool_result_memory_block(self, state: Dict[str, Any], current_turn_index: int) -> str:
        conversation = state["conversation"]
        items: List[Dict[str, Any]] = conversation.get("tool_result_memory_items", [])
        if not items:
            return ""

        eligible = [it for it in items if int(it.get("turn", -1)) < current_turn_index]
        if not eligible:
            return ""

        if self.tool_result_max_items > 0:
            eligible = eligible[-self.tool_result_max_items :]

        sep = " => " if self.memory_mode == "tool_result_memory_v2" else " -> "
        lines: List[str] = []
        for it in eligible:
            t = it.get("turn", "?")
            call = str(it.get("tool_call", ""))
            result = str(it.get("tool_result", ""))
            lines.append(f"- Turn {t} | {call}{sep}{result}")

        if self.tool_result_max_total_chars > 0:
            while lines and len("\n".join(lines)) > self.tool_result_max_total_chars:
                lines.pop(0)

        return "\n".join(lines)

    def _safe_json_obj(self, text: str) -> Dict[str, Any] | None:
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
            return None
        except Exception:
            return None

    def _observation_from_tool_result(self, tool_call: str, tool_result: str) -> str:
        """
        Convert raw tool result to a factual observation string.
        No planner hints, no extra reasoning.
        """
        call = tool_call or "unknown_tool()"
        tool_name = call.split("(", 1)[0].strip() if "(" in call else call.strip()
        result = tool_result or ""
        obj = self._safe_json_obj(result)

        # Generic error-first handling.
        if self._is_error_output(result):
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

        # Predictable fallback.
        return f"The tool {tool_name} returned: {result}"

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

        # Execute at most one tool call and append tool output for next turn context.
        env_session = state.get("env")
        tool_call, tool_result = execute_first_tool_call(env_session=env_session, response=response)
        if tool_result is not None:
            # For OpenAI chat.completions, tool messages must follow an assistant
            # message that contains the corresponding tool_calls.
            assistant_content = getattr(response.choices[0].message, "content", "") or ""
            assistant_msg: Dict[str, Any] = {"role": "assistant", "content": assistant_content}
            if tool_call is not None:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tool_call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": __import__("json").dumps(
                                tool_call.arguments, separators=(",", ":"), ensure_ascii=False
                            ),
                        },
                    }
                ]
            conversation["history_messages"].append(assistant_msg)

            # Append tool output as a tool role message for next turn.
            conversation["history_messages"].append(
                {
                    "role": "tool",
                    "content": tool_result.output,
                    # For OpenAI chat completions, tool_call_id is optional in most backends;
                    # we keep it for debugging/compat if available.
                    **({"tool_call_id": tool_result.tool_call_id} if tool_result.tool_call_id else {}),
                }
            )
            turn_index = conversation["current_turn_index"]
            if 0 <= turn_index < len(conversation.get("tool_outputs", [])):
                conversation["tool_outputs"][turn_index] = tool_result.output
            # Record lightweight tool result memory item: "tool call + result".
            selected_calls = conversation.get("history_calls", [])
            tool_call_str = ""
            if 0 <= turn_index < len(selected_calls) and selected_calls[turn_index]:
                tool_call_str = str(selected_calls[turn_index][0])
            elif tool_call is not None:
                tool_call_str = f"{tool_call.name}(...)"

            memory_items: List[Dict[str, Any]] = conversation.get("tool_result_memory_items", [])
            memory_items.append(
                {
                    "turn": turn_index,
                    "tool_call": tool_call_str,
                    "tool_result": (
                        self._observation_from_tool_result(
                            tool_call_str,
                            self._truncate_tool_output(str(tool_result.output)),
                        )
                        if self.memory_mode == "tool_result_memory_v2"
                        else self._truncate_tool_output(str(tool_result.output))
                    ),
                }
            )
            conversation["tool_result_memory_items"] = memory_items

        # Optional per-turn debug trace.
        if self.debug_mode:
            turn_index: int = conversation["current_turn_index"]
            turn_texts: List[str] = conversation.get("turn_texts", [])
            user_text = turn_texts[turn_index] if turn_index < len(turn_texts) else ""
            history_calls: List[List[str]] = conversation.get("history_calls", [])
            tools_per_turn: List[List[str]] = conversation.get("tools_per_turn", [])
            tool_outputs: List[str] = conversation.get("tool_outputs", [])
            injected_memory_block = ""

            available_tools = tools_per_turn[turn_index] if turn_index < len(tools_per_turn) else []
            selected_calls = history_calls[turn_index] if turn_index < len(history_calls) else []
            tool_output = tool_outputs[turn_index] if turn_index < len(tool_outputs) else ""

            print(f"[Bella][multi_turn_base][debug] turn={turn_index}")
            print(f"[Bella][multi_turn_base][debug] user_request={user_text!r}")
            print(f"[Bella][multi_turn_base][debug] tools={available_tools}")

            if self.memory_mode == "action_history" and turn_index > 0:
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
                print(
                    "[Bella][multi_turn_base][debug] injected_action_history=<none>"
                )
            if self.memory_mode in ("tool_result_memory", "tool_result_memory_v2"):
                injected_memory_block = self._build_tool_result_memory_block(state, turn_index)
                if injected_memory_block:
                    print(
                        f"[Bella][multi_turn_base][debug] injected_{self.memory_mode}="
                        + injected_memory_block
                    )
                else:
                    print(f"[Bella][multi_turn_base][debug] injected_{self.memory_mode}=<none>")
            else:
                print("[Bella][multi_turn_base][debug] injected_tool_result_memory=<none>")

            print(
                f"[Bella][multi_turn_base][debug] selected_calls={selected_calls}"
            )
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
        return _CATEGORIES.get(category, {}).get("result_group", "multi_turn")

    def result_filename(self, category: str) -> str:
        return _CATEGORIES.get(category, {}).get("result_filename", "BFCL_v4_multi_turn_base_result.json")

