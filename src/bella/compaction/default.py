from __future__ import annotations

from bella.types import Message
from bella.model.base import Model
from bella.memory.base import Memory
from bella.memory.react import ReactMemory, Turn, UserStep, AssistantStep
from bella.memory.user import UserMemory, UserAgentStep


REACT_COMPACTION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION for a tool-calling agent.\n"
    "Summarize the conversation and tool interactions so far.\n\n"
    "Include:\n"
    "- What the user requested and key constraints\n"
    "- Tools called, their results, and decisions made based on results\n"
    "- Current progress toward the user's goal\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data needed to continue (IDs, amounts, names, etc.)\n\n"
    "Be concise and structured. Do NOT call tools."
)

USER_COMPACTION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION for a simulated user.\n"
    "Summarize the dialogue so far from the user's perspective.\n\n"
    "Include:\n"
    "- The user's original goal and constraints\n"
    "- What information has been exchanged\n"
    "- What the agent has done or offered so far\n"
    "- What the user still needs or is waiting for\n"
    "- The user's personality and knowledge boundaries\n\n"
    "Be concise and structured."
)


class ReactDefaultCompactor:
    """Summarization-based compactor for ReactAgent.

    Calls the model to summarize the conversation history,
    then replaces all turns with a summary turn plus the last (current) turn.
    """

    def compact(self, memory: Memory, model: Model) -> ReactMemory:
        memory: ReactMemory

        messages_to_summarize = memory.to_messages(current_turn_index=-1)
        summary_messages = [
            Message(role="system", content=REACT_COMPACTION_PROMPT),
            Message(role="user", content="\n".join(
                f"[{m.role}]: {m.content or ''}" for m in messages_to_summarize
            )),
        ]

        response = model.generate(summary_messages)
        summary = response.content or "(no summary)"

        new_memory = ReactMemory(system_prompt=memory.system_prompt)
        summary_turn = Turn(
            user_step=UserStep(content=f"Context summary (auto-compacted):\n{summary}"),
            react_steps=[(AssistantStep(content="Understood, continuing."), [])],
        )
        new_memory.turns.append(summary_turn)

        if memory.turns:
            new_memory.turns.append(memory.turns[-1])

        return new_memory


class UserDefaultCompactor:
    """Summarization-based compactor for UserAgent.

    Calls the model to summarize the dialogue history,
    then replaces all steps with a summary step.
    """

    def compact(self, memory: Memory, model: Model) -> UserMemory:
        memory: UserMemory

        messages_to_summarize = memory.to_messages()
        summary_messages = [
            Message(role="system", content=USER_COMPACTION_PROMPT),
            Message(role="user", content="\n".join(
                f"[{m.role}]: {m.content or ''}" for m in messages_to_summarize
            )),
        ]

        response = model.generate(summary_messages)
        summary = response.content or "(no summary)"

        new_memory = UserMemory(system_prompt=memory.system_prompt)
        new_memory.steps.append(UserAgentStep(
            received_message=f"Context summary (auto-compacted):\n{summary}",
            generated_message="Understood, continuing.",
        ))

        return new_memory
