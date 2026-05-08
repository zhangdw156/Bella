from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from bella.types import Message, ToolCall, TokenUsage
from bella.model.base import Model, ModelAdapter


class OpenAIChatAdapter:
    def is_tool_call(self, response: Any) -> bool:
        return response.choices[0].message.tool_calls is not None

    def parse_tool_call(self, response: Any) -> list[dict]:
        return [
            {"name": tc.function.name, "arguments": json.loads(tc.function.arguments), "id": tc.id}
            for tc in response.choices[0].message.tool_calls
        ]

    def parse_reasoning(self, response: Any) -> str | None:
        return getattr(response.choices[0].message, "reasoning_content", None)

    def format_reasoning(self, content: str, reasoning_content: str) -> str:
        return content

    def role_mapping(self) -> dict[str, str]:
        return {}

    def extra_params(self) -> dict:
        return {}


class OpenAIChatModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def default_adapter(self) -> ModelAdapter:
        return OpenAIChatAdapter()

    def _map_role(self, role: str) -> str:
        mapping = self.adapter.role_mapping()
        return mapping.get(role, role)

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        result = []
        for msg in messages:
            role = self._map_role(msg.role)
            d: dict = {"role": role}

            if msg.role == "assistant":
                content = msg.content or ""
                if msg.reasoning_content:
                    content = self.adapter.format_reasoning(content, msg.reasoning_content)
                d["content"] = content

                if msg.tool_calls:
                    d["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in msg.tool_calls
                    ]
            elif msg.role == "tool":
                d["content"] = msg.content
                d["tool_call_id"] = msg.tool_call_id
            else:
                d["content"] = msg.content

            result.append(d)
        return result

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("inputSchema", {}),
                },
            }
            for t in tools
        ]

    def generate(self, messages: list[Message], tools: list[dict] | None = None) -> Message:
        kwargs: dict = {
            "model": self.model_id,
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        kwargs.update(self.adapter.extra_params())

        response = self._client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content
        reasoning_content = self.adapter.parse_reasoning(response)
        tool_calls = None
        if self.adapter.is_tool_call(response):
            parsed = self.adapter.parse_tool_call(response)
            tool_calls = [ToolCall(name=tc["name"], arguments=tc["arguments"], id=tc["id"]) for tc in parsed]

        token_usage = None
        if response.usage:
            token_usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

        return Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            token_usage=token_usage,
        )
