from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Timing:
    start_time: float
    end_time: float | None = None

    @property
    def duration(self) -> float | None:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None
    token_usage: TokenUsage | None = None
