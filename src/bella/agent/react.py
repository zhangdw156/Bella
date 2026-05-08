from __future__ import annotations

import time
from dataclasses import dataclass

from bella.types import Message, ToolCall, TokenUsage, Timing, Backend
from bella.model.base import Model
from bella.memory.react import ReactMemory, Turn, UserStep, AssistantStep, ToolResultStep
from bella.agent.base import Agent
from bella.compaction.base import ContextCompactor


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

    def init_memory(self, system_prompt: str) -> None:
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
