from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from bella.types import Message


@runtime_checkable
class ModelAdapter(Protocol):
    def is_tool_call(self, response: Any) -> bool: ...

    def parse_tool_call(self, response: Any) -> list[dict]:
        """Extract tool calls from the raw response.
        Returns: [{"name": "...", "arguments": {...}, "id": "..."}]
        """
        ...

    def parse_reasoning(self, response: Any) -> str | None: ...

    def format_reasoning(self, content: str, reasoning_content: str) -> str:
        """Format reasoning_content for feeding back into subsequent LLM calls.
        Default adapters may return content only (ignoring reasoning_content)
        if the model does not support reasoning in input.
        Custom adapters (e.g., Qwen3) wrap reasoning in model-specific tags.
        """
        ...

    def role_mapping(self) -> dict[str, str]:
        """Map internal roles to API-specific roles.
        Returns a dict like {"system": "developer"}.
        Empty dict means identity (no mapping).
        """
        ...

    def extra_params(self) -> dict:
        """Extra parameters to merge into the API call.
        E.g., {"extra_body": {"enable_thinking": True}} for Qwen3.
        """
        ...


class Model(ABC):
    def __init__(
        self,
        model_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 1.0,
        adapter: ModelAdapter | None = None,
    ):
        self.model_id = model_id
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self._adapter = adapter

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
