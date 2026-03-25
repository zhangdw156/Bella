"""
LoCoMo QA inference adapters.

Two adapters:
  ``LoCoMoQAAdapter``       – full-context baseline: entire conversation as prompt
  ``LoCoMoMemoryAdapter``   – mem0-style: ingest conversation into memory, retrieve
                              relevant facts per question (per Mem0 paper § 3)
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, List

from bella.benchmarks.base import InferAdapter
from bella.infer.types import BellaRequest, BellaResult

logger = logging.getLogger(__name__)

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

MEMORY_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to memories from past conversations. "
    "Use the provided memories to answer the question accurately. "
    "Answer with a short phrase. If the memories don't contain enough "
    "information, say 'No information available'."
)

MEMORY_QA_PROMPT = (
    "Relevant memories from the conversation:\n{memories}\n\n"
    "Based on the memories above, write an answer in the form of a short "
    "phrase for the following question.\n\n"
    "Question: {question}\nShort answer:"
)


# ── shared helpers ───────────────────────────────────────────────────

def _format_conversation(conversation: Dict[str, Any], max_chars: int = 0) -> str:
    """Format LoCoMo conversation sessions into a readable context string."""
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


def _parse_qa_response(entry: Dict[str, Any], response: Any) -> BellaResult:
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


# ── full-context adapter (baseline) ─────────────────────────────────

class LoCoMoQAAdapter(InferAdapter):
    """Full-context baseline: entire conversation as prompt context."""

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

    def parse_response(self, entry: Dict[str, Any], response: Any, state: Dict[str, Any]) -> BellaResult:
        return _parse_qa_response(entry, response)


# ── memory-augmented adapter (per Mem0 paper) ────────────────────────

class LoCoMoMemoryAdapter(InferAdapter):
    """Memory-augmented QA: ingest conversation into memory, retrieve per question.

    Follows the Mem0 paper (Chhikara et al., 2025) evaluation protocol:
    1. on_run_start: open memory session
    2. First time a conversation is seen: ingest all turns via memory.add()
    3. build_request: memory.search(question) → retrieved facts as context
    4. on_run_end: close and clear memory
    """

    def __init__(self) -> None:
        from bella.memory import create_memory
        self._memory = create_memory()
        self._ingested: set[str] = set()
        self._ingest_lock = threading.Lock()
        self._max_memories = int(os.getenv("BELLA_MEM0_MAX_RESULTS", "10"))

    def on_run_start(self, session_id: str) -> None:
        self._memory.open(session_id)
        self._ingested.clear()

    def on_run_end(self) -> None:
        self._memory.close()
        self._ingested.clear()

    def _ingest_conversation(self, sample_id: str, conversation: Dict[str, Any]) -> None:
        """Ingest all turns of a conversation into memory (once per conversation)."""
        with self._ingest_lock:
            if sample_id in self._ingested:
                return
            self._ingested.add(sample_id)

        logger.info("Ingesting conversation %s into memory", sample_id)
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
            date_time = conversation.get(dt_key, "")
            for turn in conversation[key]:
                speaker = turn.get("speaker", "")
                text = turn.get("text", "")
                content = f'{speaker} said, "{text}"'
                if "blip_caption" in turn:
                    content += f' and shared {turn["blip_caption"]}'
                if date_time:
                    content = f"[{date_time}] {content}"
                self._memory.add(content, {"session": num, "speaker": speaker})
        logger.info("Ingested conversation %s (%d sessions)", sample_id, len(session_nums))

    def init_state(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        self._ingest_conversation(entry["sample_id"], entry["conversation"])
        return {}

    def build_request(self, entry: Dict[str, Any], state: Dict[str, Any]) -> BellaRequest:
        question = entry["question"]
        memories = self._memory.search(question, limit=self._max_memories)

        if memories:
            memory_text = "\n".join(f"- {m}" for m in memories)
            user_content = MEMORY_QA_PROMPT.format(memories=memory_text, question=question)
        else:
            conversation = entry["conversation"]
            speaker_a, speaker_b = _get_speakers(conversation)
            header = CONV_HEADER.format(speaker_a=speaker_a, speaker_b=speaker_b)
            context = _format_conversation(conversation, max_chars=8000)
            qa_prompt = QA_PROMPT.format(question=question)
            user_content = header + context + "\n\n" + qa_prompt

        system = MEMORY_SYSTEM_PROMPT if memories else SYSTEM_PROMPT
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        return BellaRequest(messages=messages, tools=[], temperature=0.0)

    def parse_response(self, entry: Dict[str, Any], response: Any, state: Dict[str, Any]) -> BellaResult:
        return _parse_qa_response(entry, response)
