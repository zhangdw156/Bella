from __future__ import annotations

from dataclasses import dataclass

from bella.types import Message
from bella.memory.base import Memory


@dataclass
class UserAgentStep:
    received_message: str
    generated_message: str
    reasoning_content: str | None = None
    is_done: bool = False


class UserMemory(Memory):
    def __init__(self, system_prompt: str):
        super().__init__(system_prompt)
        self.steps: list[UserAgentStep] = []

    def to_messages(self) -> list[Message]:
        """Reconstruct messages with role inversion.

        system: persona + demand + rules
        assistant: generated user messages (UserAgent generates "user" text via assistant role)
        user: received assistant responses (real assistant responses fed as "user" role)
        """
        messages: list[Message] = [Message(role="system", content=self.system_prompt)]

        for step in self.steps:
            if step.received_message:
                messages.append(Message(role="user", content=step.received_message))
            messages.append(Message(role="assistant", content=step.generated_message))

        return messages

    def estimate_tokens(self) -> int:
        total_chars = len(self.system_prompt)
        for step in self.steps:
            total_chars += len(step.received_message)
            total_chars += len(step.generated_message)
        return -(-total_chars // 4)
