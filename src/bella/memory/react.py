from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from bella.types import Message, Timing, ToolCall, TokenUsage
from bella.memory.base import Memory


@dataclass
class UserStep:
    content: str


@dataclass
class AssistantStep:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_content: str | None = None
    token_usage: TokenUsage | None = None


@dataclass
class ToolResultStep:
    tool_call_id: str
    name: str
    result: dict[str, Any]


@dataclass
class Turn:
    user_step: UserStep
    react_steps: list[tuple[AssistantStep, list[ToolResultStep]]] = field(default_factory=list)
    timing: Timing | None = None


class ReactMemory(Memory):
    def __init__(self, system_prompt: str):
        super().__init__(system_prompt)
        self.turns: list[Turn] = []

    def to_messages(self, *, current_turn_index: int) -> list[Message]:
        """Reconstruct messages from Steps.

        Reasoning visibility rule:
        - Previous turns (index < current_turn_index): reasoning_content STRIPPED
        - Current turn (index == current_turn_index): reasoning_content INCLUDED
        """
        messages: list[Message] = [Message(role="system", content=self.system_prompt)]

        for i, turn in enumerate(self.turns):
            include_reasoning = (i == current_turn_index)

            messages.append(Message(role="user", content=turn.user_step.content))

            for assistant_step, tool_results in turn.react_steps:
                messages.append(Message(
                    role="assistant",
                    content=assistant_step.content,
                    tool_calls=list(assistant_step.tool_calls) if assistant_step.tool_calls else None,
                    reasoning_content=assistant_step.reasoning_content if include_reasoning else None,
                ))

                for tool_result in tool_results:
                    messages.append(Message(
                        role="tool",
                        content=json.dumps(tool_result.result, ensure_ascii=False),
                        tool_call_id=tool_result.tool_call_id,
                    ))

        return messages

    def estimate_tokens(self) -> int:
        total_chars = len(self.system_prompt)
        for turn in self.turns:
            total_chars += len(turn.user_step.content)
            for assistant_step, tool_results in turn.react_steps:
                total_chars += len(assistant_step.content or "")
                if assistant_step.reasoning_content:
                    total_chars += len(assistant_step.reasoning_content)
                for tool_result in tool_results:
                    total_chars += len(json.dumps(tool_result.result, ensure_ascii=False))
        return -(-total_chars // 4)
