from __future__ import annotations

from dataclasses import dataclass

from bella.types import Message
from bella.model.base import Model
from bella.memory.user import UserMemory, UserAgentStep
from bella.agent.base import Agent
from bella.compaction.base import ContextCompactor


DONE_MARKER = "[DONE]"

_SYSTEM_TEMPLATE = """\
You simulate a real user in a multi-turn dialogue with a customer service assistant.

## Your Persona
Role: {role}
Personality: {personality}
Knowledge boundary: {knowledge_boundary}

## Your Goal
{demand}

## Rules
- Stay fully in character. Use vocabulary and tone natural to your persona.
- Do NOT use tool names, API endpoints, parameter names, or system internals.
- Respond to the assistant's questions based on what your persona would know.
- If the assistant asks for information outside your knowledge boundary, say you don't know.
- When the assistant has fully completed your goal, output exactly: {done_marker}
- If the assistant cannot do what you asked and there is no way forward, output: {done_marker}
- Do NOT output {done_marker} until the goal is genuinely resolved or clearly impossible.
- Keep messages to 1-3 sentences. No markdown, no bullet points, no labels.

## Critical: Reveal Information Gradually
- In your FIRST message, only mention your primary reason for calling and give just \
enough context for the assistant to start helping (e.g. your name, that you need help \
with a booking). Do NOT list all your requests upfront.
- Reveal additional requests and specific details (dates, names, amounts) only when the \
assistant asks or after the current issue is resolved.
- A real person does not open a call by listing every single thing they need in one \
breath — they start with the most important thing and bring up the rest as the \
conversation progresses.
"""


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
        return _SYSTEM_TEMPLATE.format(
            role=user_agent_config.get("role", "a user"),
            personality=user_agent_config.get("personality", ""),
            knowledge_boundary=user_agent_config.get("knowledge_boundary", ""),
            demand=demand,
            done_marker=DONE_MARKER,
        )

    def _parse_response(self, response: Message) -> UserTurnResult:
        raw = response.content or ""
        is_done = DONE_MARKER in raw
        cleaned = raw.replace(DONE_MARKER, "").strip()
        return UserTurnResult(message=cleaned, is_done=is_done)

    def start(self, demand: str, user_agent_config: dict) -> UserTurnResult:
        """Initialize memory and generate the first user message."""
        system_prompt = self._build_system_prompt(demand, user_agent_config)
        self.memory = UserMemory(system_prompt=system_prompt)

        messages = self.memory.to_messages()
        messages.append(Message(
            role="user",
            content=(
                "The conversation is starting now. Generate your first "
                "message to the assistant. Mention only your most "
                "immediate need — do not list everything upfront."
            ),
        ))

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
