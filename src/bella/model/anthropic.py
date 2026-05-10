from __future__ import annotations

import json

from anthropic import Anthropic

from bella.types import Message, ToolCall, TokenUsage
from bella.model.base import Model, ModelAdapter


class AnthropicAdapter:
    def parse_reasoning(self, content: str) -> tuple[str, str | None]:
        return content, None

    def parse_tool_call(self, content: str) -> tuple[str, list[dict] | None]:
        return content, None

    def format_reasoning(self, content: str, reasoning_content: str) -> str:
        return content

    def role_mapping(self) -> dict[str, str]:
        return {}

    def extra_params(self) -> dict:
        return {}


class AnthropicModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = Anthropic(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    def default_adapter(self) -> ModelAdapter:
        return AnthropicAdapter()

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict]]:
        """Returns (system_prompt, messages). Anthropic takes system separately."""
        system_prompt = ""
        result = []

        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content or ""
                continue

            if msg.role == "assistant":
                content_parts = []
                content = msg.content or ""
                if msg.reasoning_content:
                    content = self.adapter.format_reasoning(content, msg.reasoning_content)
                if content:
                    content_parts.append({"type": "text", "text": content})

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content_parts.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })

                if content_parts:
                    result.append({"role": "assistant", "content": content_parts})

            elif msg.role == "tool":
                if result and result[-1]["role"] == "user":
                    result[-1]["content"].append({
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content or "",
                    })
                else:
                    result.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content or "",
                        }],
                    })

            elif msg.role == "user":
                result.append({"role": "user", "content": [{"type": "text", "text": msg.content or "(no content)"}]})

        return system_prompt, result

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("inputSchema", {}),
            }
            for t in tools
        ]

    def generate(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        system_prompt, api_messages = self._convert_messages(messages)

        kwargs: dict = {
            "model": self.model_id,
            "system": system_prompt,
            "messages": api_messages,
            "temperature": self.temperature,
            "max_tokens": 4096,
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        kwargs.update(self.adapter.extra_params())

        response = self._client.messages.create(**kwargs)

        # Protocol-level extraction
        text_parts = [block.text.strip() for block in response.content if block.type == "text" and block.text.strip()]
        content = "\n".join(text_parts) if text_parts else ""
        thinking = [block.thinking for block in response.content if block.type == "thinking"]
        sdk_reasoning = "\n".join(thinking) if thinking else None
        sdk_tool_calls = [block for block in response.content if block.type == "tool_use"]

        # Adapter fallback for reasoning
        if sdk_reasoning:
            reasoning_content = sdk_reasoning
        else:
            content, reasoning_content = self.adapter.parse_reasoning(content)

        # Adapter fallback for tool calls
        tool_calls = None
        if sdk_tool_calls:
            tool_calls = [
                ToolCall(name=block.name, arguments=block.input, id=block.id)
                for block in sdk_tool_calls
            ]
        else:
            content, parsed_tool_calls = self.adapter.parse_tool_call(content)
            if parsed_tool_calls:
                tool_calls = [ToolCall(name=tc["name"], arguments=tc["arguments"], id=tc["id"]) for tc in parsed_tool_calls]

        token_usage = None
        if response.usage:
            token_usage = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        return Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            token_usage=token_usage,
        )
