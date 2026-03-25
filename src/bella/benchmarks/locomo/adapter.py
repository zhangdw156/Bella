"""
LoCoMo QA inference adapter.

Builds a conversation-context + question prompt, sends it to the LLM
(no tool calling), and parses the free-text answer.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

from bella.benchmarks.base import InferAdapter
from bella.infer.types import BellaRequest, BellaResult

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about conversations. "
    "Answer with exact words from the conversation whenever possible. "
    "Keep your answer as a short phrase."
)

CONV_HEADER = (
    "Below is a conversation between two people: {speaker_a} and {speaker_b}. "
    "The conversation takes place over multiple days and the date of each "
    "conversation is written at the beginning of the conversation.\n\n"
)

QA_PROMPT = (
    "Based on the above context, write an answer in the form of a short "
    "phrase for the following question. Answer with exact words from the "
    "context whenever possible.\n\n"
    "Question: {question}\nShort answer:"
)

QA_PROMPT_ADVERSARIAL = (
    "Based on the above context, answer the following question.\n\n"
    "Question: {question}\nShort answer:"
)


def _format_conversation(conversation: Dict[str, Any], max_chars: int = 0) -> str:
    """Format LoCoMo conversation sessions into a readable context string.

    Iterates sessions in chronological order.  If *max_chars* > 0 the context
    is truncated from the **earliest** sessions to fit.
    """
    session_blocks: List[str] = []
    session_nums = sorted(
        int(k.split("_")[1])
        for k in conversation
        if k.startswith("session_") and "date_time" not in k
    )

    for num in session_nums:
        key = f"session_{num}"
        dt_key = f"session_{num}_date_time"
        if key not in conversation or not conversation[key]:
            continue

        lines: List[str] = []
        date_time = conversation.get(dt_key, "")
        for turn in conversation[key]:
            speaker = turn.get("speaker", "")
            text = turn.get("text", "")
            line = f'{speaker} said, "{text}"'
            if "blip_caption" in turn:
                line += f' and shared {turn["blip_caption"]}'
            lines.append(line)

        block = ""
        if date_time:
            block += f"DATE: {date_time}\nCONVERSATION:\n"
        block += "\n".join(lines)
        session_blocks.append(block)

    full = "\n\n".join(session_blocks)

    if max_chars > 0 and len(full) > max_chars:
        while session_blocks and len("\n\n".join(session_blocks)) > max_chars:
            session_blocks.pop(0)
        full = "\n\n".join(session_blocks)

    return full


def _get_speakers(conversation: Dict[str, Any]) -> tuple[str, str]:
    """Extract speaker names from the first session."""
    speakers: set[str] = set()
    for key in conversation:
        if key.startswith("session_") and "date_time" not in key and conversation[key]:
            for turn in conversation[key]:
                speakers.add(turn.get("speaker", "Unknown"))
                if len(speakers) >= 2:
                    break
        if len(speakers) >= 2:
            break
    names = sorted(speakers)
    return (names[0] if len(names) > 0 else "A", names[1] if len(names) > 1 else "B")


class LoCoMoQAAdapter(InferAdapter):
    """Single-turn QA adapter: conversation context + question → short answer."""

    def __init__(self) -> None:
        self._max_context_chars = int(
            os.getenv("BELLA_LOCOMO_MAX_CONTEXT_CHARS", "0")
        )

    def build_request(self, entry: Dict[str, Any], state: Dict[str, Any]) -> BellaRequest:
        conversation = entry["conversation"]
        question = entry["question"]
        category = entry.get("category", 0)

        speaker_a, speaker_b = _get_speakers(conversation)
        header = CONV_HEADER.format(speaker_a=speaker_a, speaker_b=speaker_b)
        context = _format_conversation(conversation, self._max_context_chars)

        if category == 5:
            qa_prompt = QA_PROMPT_ADVERSARIAL.format(question=question)
        else:
            qa_prompt = QA_PROMPT.format(question=question)

        user_content = header + context + "\n\n" + qa_prompt

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        return BellaRequest(messages=messages, tools=[], temperature=0.0)

    def parse_response(
        self,
        entry: Dict[str, Any],
        response: Any,
        state: Dict[str, Any],
    ) -> BellaResult:
        content = getattr(response.choices[0].message, "content", "") or ""
        prediction = content.strip()

        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        return BellaResult(
            id=entry["id"],
            result=[{"prediction": prediction}],
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            latency=0.0,
            extra={
                "question": entry["question"],
                "gold_answer": entry["answer"],
                "category": entry["category"],
            },
        )
