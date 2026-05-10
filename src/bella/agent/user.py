from __future__ import annotations

from dataclasses import dataclass

from bella.types import Message
from bella.model.base import Model
from bella.memory.user import UserMemory, UserAgentStep
from bella.agent.base import Agent
from bella.compaction.base import ContextCompactor


DONE_MARKER = "[DONE]"

_SYSTEM_TEMPLATE = """\
You simulate a real user in a multi-turn dialogue with an AI assistant.

## Your Persona
Role: {role}
Personality: {personality}
Knowledge boundary: {knowledge_boundary}

## Your Goal
{demand}

## Rules
- Stay fully in character. Use vocabulary and tone natural to your persona.
- Do NOT use tool names, API endpoints, parameter names, or system internals.
- When the assistant directly asks for information within your knowledge boundary, answer immediately.
- Do not volunteer information the assistant has not asked for, but never withhold directly requested information that you know.
- If the assistant misunderstands your request or acts on wrong information, correct it clearly.
- When the assistant lists action details and asks for your confirmation, review them against your goal. Confirm if correct; point out discrepancies if not.
- If the assistant asks for information outside your knowledge boundary, say you don't know or give a vague answer.
- Keep messages to 1-3 sentences. No markdown, no bullet points, no labels.

## Completing the Conversation
- When the assistant has confirmed completing ALL parts of your goal with specific results (e.g. confirmation numbers, status changes), say a brief closing remark and output: {done_marker}
- If the assistant clearly states it cannot help and there is no alternative path, output: {done_marker}
- Do NOT output {done_marker} if the assistant merely acknowledged your request without completing it.
- Do NOT continue the conversation after all goals are genuinely resolved.

## Reveal Information Gradually
- In your FIRST message, only describe your most immediate need and give just enough context for the assistant to start helping (e.g. your name, what you need help with). Do NOT list all your requests upfront.
- Bring up secondary goals only after the current goal is resolved or naturally relevant.
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
