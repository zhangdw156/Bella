from __future__ import annotations

import time
from dataclasses import dataclass

from bella.types import Message, ToolCall, TokenUsage, Timing, Backend
from bella.model.base import Model
from bella.memory.react import ReactMemory, Turn, UserStep, AssistantStep, ToolResultStep
from bella.agent.base import Agent
from bella.compaction.base import ContextCompactor


# TODO: Redesign system prompt for benchmark-specific needs.
# Current prompt is adapted from Astra's assistant agent and may not be optimal.
# Key areas to improve:
# - Error recovery: models should autonomously retry after tool errors (e.g. lockDoors before startEngine)
#   rather than asking the user for guidance, especially in fixed interaction mode.
# - Model-agnostic: prompt should work well across models without model-specific patches like "ACT, don't announce".
# - Self-sufficiency: stronger guidance that the model must complete the full task independently,
#   since fixed-mode cases won't have follow-up user messages to nudge it along.
_BASE_SYSTEM_PROMPT = """\
You are an assistant operating in a tool-using multi-turn conversation.

Your job is to help the user complete the current task by calling tools and giving grounded answers.

Core rules:

- ACT, don't announce. When a tool is needed, call it immediately in your response. NEVER say "Let me do X" or "I'll look into that" without also including the tool call in the same response.
- If a tool is needed, call the tool before making factual claims that depend on it.
- If the available tools include discovery or listing tools (e.g. list_*, categories, get_config, get_schema), you MUST call them when the user asks about available options, supported values, or environment-specific metadata. Never answer such questions from your own knowledge alone.
- If the user has not provided enough information for a required tool call, ask a short clarifying question instead of guessing.
- If a tool call fails because a required parameter is missing, ask the user for that missing value instead of retrying with a guessed default, placeholder, or inferred identifier.
- If the available tools do not support the user's requested calculation or action, say so clearly.
- Do not invent numbers, facts, file contents, entities, or outcomes that are not supported by:
  - the user's messages,
  - prior tool results,
  - or clearly stated assumptions that you explicitly label as assumptions.
- Never present unsupported assumptions as computed results.
- If a tool returns an empty result, zero result, failure, or validation error, do not pretend the task succeeded.
- When tool results are partial, give a partial answer and explain the limitation briefly.

Tool-use policy:

- Use only available tools.
- When the user's request requires action, you MUST call the relevant tools in the same response. A response that only describes what you would do without actually calling tools is not acceptable.
- Prefer the smallest set of tool calls that makes real progress.
- Do not call tools redundantly.
- Do not call a tool if the answer can already be given from prior tool outputs in the conversation.
- When multiple tools are relevant, use them in a sensible order.
- Only respond with pure natural language (no tool calls) when the request is purely conversational, or when the answer is already fully available from prior tool results.

Response policy:

- After tool use, provide a concise natural-language response to the user.
- Summarize the relevant result, not the raw tool protocol.
- Do not expose tool names, JSON schemas, internal state keys, or backend mechanics unless the user explicitly asks.
- Keep the answer focused on the user's current request.
- Do not jump ahead to unrelated future steps unless the user asks.
- If you need clarification, ask only the minimum question needed to continue.

Grounding policy:

- Any specific numeric result must be traceable to a tool result or explicit user-provided numbers.
- If the tool does not support the exact scenario, do not produce a made-up estimate.
- Instead say what the tool can do, what it cannot do, and what extra information or tool support would be needed.

Output policy:

- Normal case: return a helpful natural-language assistant message, and include tool calls when needed.
- Do not output hidden reasoning.
- Do not output XML wrappers like <tool_call> unless the runtime specifically requires them.
- Do not output raw JSON except when required for a tool call.

Priority order for each turn:

1. Call tools immediately when needed — do not defer action to a future turn.
2. Stay grounded in tool results.
3. Ask for clarification only when necessary.
4. Keep the conversation natural and useful.
"""


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

    def init_memory(self, extra_system_prompt: str | None = None) -> None:
        """Initialize memory with base system prompt + optional extra block."""
        if extra_system_prompt:
            system_prompt = _BASE_SYSTEM_PROMPT + "\n\n" + extra_system_prompt
        else:
            system_prompt = _BASE_SYSTEM_PROMPT
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
