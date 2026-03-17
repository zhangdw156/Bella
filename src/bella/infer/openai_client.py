from __future__ import annotations

from typing import Any, Iterable, List, Mapping, MutableMapping

from openai import OpenAI

from bella.config import load_settings


class OpenAIClient:
    """
    Minimal OpenAI-compatible client wrapper used by Bella.

    Responsibilities:
    - Read API key / base URL / model from Bella's .env (via load_settings)
    - Expose a small `chat_with_tools` API that supports function/tool calling.
    """

    def __init__(self) -> None:
        settings = load_settings()

        client_kwargs: dict[str, Any] = {
            "api_key": settings.openai_api_key,
        }
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url

        self._client = OpenAI(**client_kwargs)
        self._model = settings.openai_model

    @property
    def model(self) -> str:
        return self._model

    def chat_with_tools(
        self,
        messages: List[Mapping[str, Any]],
        tools: Iterable[Mapping[str, Any]] | None = None,
        temperature: float = 0.0,
    ):
        """
        Call an OpenAI-compatible /chat/completions endpoint with optional tools.
        """
        tools_list: List[Mapping[str, Any]] | None = list(tools) if tools else None

        kwargs: MutableMapping[str, Any] = {
            "model": self._model,
            "messages": list(messages),
            "temperature": float(temperature),
            "store": False,
        }
        if tools_list:
            kwargs["tools"] = tools_list
            kwargs["tool_choice"] = "auto"

        return self._client.chat.completions.create(**kwargs)

