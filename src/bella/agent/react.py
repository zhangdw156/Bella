from __future__ import annotations

import time
from dataclasses import dataclass

from bella.types import Message, ToolCall, TokenUsage, Timing, Backend
from bella.model.base import Model
from bella.memory.react import ReactMemory, Turn, UserStep, AssistantStep, ToolResultStep
from bella.agent.base import Agent
from bella.compaction.base import ContextCompactor


_COMMON_BLOCK = """\
You are an assistant operating in a tool-using multi-turn conversation.

Your job is to help the user complete the current task by calling tools and giving grounded answers.

Tool-use policy:

- Use only available tools.
- When the user's request requires action, call the relevant tools in the same response. A response that only describes what you would do without calling tools is not acceptable.
- If the available tools include discovery or listing tools (e.g. list_*, get_config, get_schema), call them when the user asks about available options or environment-specific metadata. Never answer such questions from your own knowledge alone.
- Prefer the smallest set of tool calls that makes real progress.
- Do not call tools redundantly or repeat a call whose result is already available.

Grounding policy:

- Do not invent numbers, facts, file contents, entities, or outcomes not supported by the user's messages, prior tool results, or clearly labeled assumptions.
- Any specific numeric result must be traceable to a tool result or user-provided data.
- If a tool returns an empty result, failure, or validation error, do not pretend the task succeeded.
- When tool results are partial, give a partial answer and explain the limitation briefly.

Output policy:

- After tool use, provide a concise natural-language response to the user.
- Summarize the relevant result, not the raw tool protocol.
- Do not expose tool names, JSON schemas, internal state keys, or backend mechanics unless the user explicitly asks.
- Do not output hidden reasoning or XML wrappers like <tool_call> unless the runtime specifically requires them."""

_FIXED_BEHAVIOR = """\
Behavioral rules (non-interactive mode):

- You must complete the user's request fully and independently within each turn. No follow-up messages will be sent to guide you.
- Never ask clarifying questions. Use available tools to discover any information you need.
- If a tool call fails, analyze the error message and take corrective action autonomously. For example, if an action requires a precondition (e.g. locking doors before starting an engine), fulfill the precondition and retry — do not report the error and stop.
- When the user's request implies a sequence of dependent actions, execute the full sequence without waiting for step-by-step confirmation.
- After completing a sub-task, continue with remaining parts of the request proactively."""

_DYNAMIC_BEHAVIOR = """\
Behavioral rules (interactive mode):

- Always include tool calls in the same response where they are needed. Do not say "I will do X" without actually calling the tool.
- If the user has not provided enough information for a required tool parameter, ask a short clarifying question.
- If a tool call fails, first try to resolve the issue autonomously by analyzing the error. Only ask the user for help if you cannot determine the correct action from tool results and context.
- After completing a sub-task, report the result and let the user direct the next step.
- Keep the conversation focused on the user's current request. Do not jump ahead to unrelated future steps unless the user asks."""

_DOMAIN_POLICY_TRANSITION = """\
If a domain policy follows, it defines your specific role and operational constraints. Where it conflicts with the rules above, the domain policy takes precedence."""

_FIXED_SYSTEM_PROMPT = _COMMON_BLOCK + "\n\n" + _FIXED_BEHAVIOR + "\n\n" + _DOMAIN_POLICY_TRANSITION
_DYNAMIC_SYSTEM_PROMPT = _COMMON_BLOCK + "\n\n" + _DYNAMIC_BEHAVIOR + "\n\n" + _DOMAIN_POLICY_TRANSITION


@dataclass
class TurnResult:
    assistant_message: str
    tool_calls: list[ToolCall]
    reasoning_content: str | None
    token_usage: TokenUsage | None
    timing: Timing | None


class ReactAgent(Agent):
    def __init__(
        self,
        model: Model,
        max_llm_calls_per_turn: int = 12,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.max_llm_calls_per_turn = max_llm_calls_per_turn
        self.memory: ReactMemory | None = None

    def default_compactor(self) -> ContextCompactor:
        from bella.compaction.default import ReactDefaultCompactor
        return ReactDefaultCompactor()

    def init_memory(self, interaction_mode: str, extra_system_prompt: str | None = None) -> None:
        """Initialize memory with mode-specific system prompt + optional domain policy."""
        base = _FIXED_SYSTEM_PROMPT if interaction_mode == "fixed" else _DYNAMIC_SYSTEM_PROMPT
        if extra_system_prompt:
            system_prompt = base + "\n\n" + extra_system_prompt
        else:
            system_prompt = base
        self.memory = ReactMemory(system_prompt=system_prompt)

    def run_turn(self, user_message: str, tools: list[dict], backend: Backend) -> TurnResult:
        """Execute one conversation turn (may involve multiple LLM calls)."""
        assert self.memory is not None

        turn = Turn(user_step=UserStep(content=user_message))
        self.memory.turns.append(turn)
        current_turn_idx = len(self.memory.turns) - 1

        start_time = time.time()
        all_tool_calls: list[ToolCall] = []
        total_token_usage: TokenUsage | None = None
        last_content = ""
        last_reasoning = None

        for _ in range(self.max_llm_calls_per_turn):
            if self.memory.estimate_tokens() > self.model.compaction_threshold:
                self.memory = self.compactor.compact(self.memory, self.model)
                current_turn_idx = len(self.memory.turns) - 1

            messages = self.memory.to_messages(current_turn_index=current_turn_idx)
            response: Message = self.model.generate(messages, tools)

            content = response.content or ""
            reasoning_content = response.reasoning_content
            last_content = content
            last_reasoning = reasoning_content

            if response.token_usage:
                if total_token_usage is None:
                    total_token_usage = response.token_usage
                else:
                    total_token_usage = TokenUsage(
                        input_tokens=total_token_usage.input_tokens + response.token_usage.input_tokens,
                        output_tokens=total_token_usage.output_tokens + response.token_usage.output_tokens,
                    )

            if response.tool_calls:
                tool_results = []
                for tc in response.tool_calls:
                    result = backend.call(tc.name, tc.arguments)
                    tool_results.append(ToolResultStep(
                        tool_call_id=tc.id, name=tc.name, result=result
                    ))

                assistant_step = AssistantStep(
                    content=content,
                    tool_calls=list(response.tool_calls),
                    reasoning_content=reasoning_content,
                    token_usage=response.token_usage,
                )
                turn.react_steps.append((assistant_step, tool_results))
                all_tool_calls.extend(response.tool_calls)
            else:
                assistant_step = AssistantStep(
                    content=content,
                    tool_calls=[],
                    reasoning_content=reasoning_content,
                    token_usage=response.token_usage,
                )
                turn.react_steps.append((assistant_step, []))
                break

        end_time = time.time()
        turn.timing = Timing(start_time=start_time, end_time=end_time)

        return TurnResult(
            assistant_message=last_content,
            tool_calls=all_tool_calls,
            reasoning_content=last_reasoning,
            token_usage=total_token_usage,
            timing=Timing(start_time=start_time, end_time=end_time),
        )
