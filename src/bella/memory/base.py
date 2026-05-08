from __future__ import annotations

from abc import ABC, abstractmethod

from bella.types import Message


class Memory(ABC):
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt

    @abstractmethod
    def to_messages(self, **kwargs) -> list[Message]: ...

    @abstractmethod
    def estimate_tokens(self) -> int: ...
