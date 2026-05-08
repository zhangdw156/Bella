from __future__ import annotations

from dataclasses import dataclass

from bella.types import Message
from bella.model.base import Model
from bella.memory.user import UserMemory, UserAgentStep
from bella.agent.base import Agent
from bella.compaction.base import ContextCompactor


DONE_SIGNAL = "[DONE]"


@dataclass
class UserTurnResult:
    message: str
    is_done: bool


class UserAgent(Agent):
    def __init__(
        self,
        model: Model,
        max_turns: int = 30,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        self.max_turns = max_turns
        self.memory: UserMemory | None = None

    def default_compactor(self) -> ContextCompactor:
        from bella.compaction.default import UserDefaultCompactor
        return UserDefaultCompactor()

    def _build_system_prompt(self, demand: str, user_agent_config: dict) -> str:
        role = user_agent_config.get("role", "")
        personality = user_agent_config.get("personality", "")
        knowledge_boundary = user_agent_config.get("knowledge_boundary", "")

        return (
            f"You are simulating a user interacting with a customer service agent.\n\n"
            f"## Your Role\n{role}\n\n"
            f"## Your Personality\n{personality}\n\n"
            f"## What You Know\n{knowledge_boundary}\n\n"
            f"## Your Goal\n{demand}\n\n"
            f"## Rules\n"
            f"- Reveal information gradually. Your first message should state only your primary need.\n"
            f"- If asked about something you don't know, say you don't know or give a vague answer.\n"
            f"- When your goal is achieved, impossible, or the agent transfers you to a human, "
            f"end your message with {DONE_SIGNAL}\n"
            f"- Stay in character. Do not break the fourth wall.\n"
        )

    def _parse_response(self, response: Message) -> UserTurnResult:
        content = response.content or ""
        is_done = DONE_SIGNAL in content
        message = content.replace(DONE_SIGNAL, "").strip()
        return UserTurnResult(message=message, is_done=is_done)

    def start(self, demand: str, user_agent_config: dict) -> UserTurnResult:
        """Initialize memory and generate the first user message."""
        system_prompt = self._build_system_prompt(demand, user_agent_config)
        self.memory = UserMemory(system_prompt=system_prompt)

        messages = self.memory.to_messages()
        messages.append(Message(role="user", content="Begin the conversation. State your need."))

        response = self.model.generate(messages)

        result = self._parse_response(response)
        self.memory.steps.append(UserAgentStep(
            received_message="",
            generated_message=result.message,
            reasoning_content=response.reasoning_content,
            is_done=result.is_done,
        ))
        return result

    def respond(self, assistant_message: str) -> UserTurnResult:
        """Generate the next user message in response to the assistant."""
        assert self.memory is not None

        if self.memory.estimate_tokens() > self.model.compaction_threshold:
            self.memory = self.compactor.compact(self.memory, self.model)

        messages = self.memory.to_messages()
        messages.append(Message(role="user", content=assistant_message))

        response = self.model.generate(messages)

        result = self._parse_response(response)
        self.memory.steps.append(UserAgentStep(
            received_message=assistant_message,
            generated_message=result.message,
            reasoning_content=response.reasoning_content,
            is_done=result.is_done,
        ))
        return result
