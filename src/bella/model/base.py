from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from bella.types import Message


@runtime_checkable
class ModelAdapter(Protocol):
    def parse_reasoning(self, content: str) -> tuple[str, str | None]:
        """Fallback: extract reasoning from content when SDK-level extraction fails.
        Returns (cleaned_content, reasoning_content).
        Inverse of format_reasoning.
        """
        ...

    def parse_tool_call(self, content: str) -> tuple[str, list[dict] | None]:
        """Fallback: extract tool calls from content when SDK-level extraction fails.
        Returns (cleaned_content, tool_calls).
        tool_calls format: [{"name": "...", "arguments": {...}, "id": "..."}]
        """
        ...

    def format_reasoning(self, content: str, reasoning_content: str) -> str:
        """Combine content and reasoning_content for feeding back to the model.
        Inverse of parse_reasoning.
        """
        ...

    def role_mapping(self) -> dict[str, str]:
        """Map internal roles to API-specific roles.
        Returns a dict like {"system": "developer"}.
        Empty dict means identity (no mapping).
        """
        ...

    def extra_params(self) -> dict:
        """Extra parameters to merge into the API call."""
        ...


_COMPACTION_RATIO = 0.8


class Model(ABC):
    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 1.0,
        max_context_tokens: int = 128000,
        adapter: ModelAdapter | None = None,
    ):
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_context_tokens = max_context_tokens
        self._adapter = adapter

    @property
    def compaction_threshold(self) -> int:
        return int(self.max_context_tokens * _COMPACTION_RATIO)

    @property
    def adapter(self) -> ModelAdapter:
        if self._adapter is not None:
            return self._adapter
        return self.default_adapter()

    @abstractmethod
    def default_adapter(self) -> ModelAdapter: ...

    @abstractmethod
    def _convert_tools(self, tools: list[dict]) -> list[dict]: ...

    @abstractmethod
    def generate(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        """Generate a response from the model.

        Args:
            messages: Conversation history in internal format.
            tools: Tool definitions from tools.jsonl.

        Returns:
            Message with content, tool_calls, reasoning_content, token_usage filled.
        """
        ...

    def to_config(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_class": type(self).__name__,
            "adapter_class": type(self.adapter).__name__,
            "max_context_tokens": self.max_context_tokens,
            "temperature": self.temperature,
        }
