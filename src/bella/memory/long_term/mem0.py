"""
Self-contained mem0-style memory plugin with global disk persistence.

Implements the core mem0 workflow — LLM fact extraction + embedding + cosine
retrieval — using only the OpenAI SDK already in the project.

Memory is **global and persistent**: all BFCL entries within a run (and across
runs) share the same store on disk.  As the system processes more entries it
accumulates experience, making later tool-call decisions better informed.

Storage format:
  JSONL file  (one JSON object per line: {"text": "...", "embedding": [...]})
  loaded into RAM on startup; new entries are appended and flushed immediately.

Search is numpy-accelerated (vectorised cosine similarity).

Environment variables (all optional):
  BELLA_MEM0_STORE_PATH         – Path to the JSONL store file
                                  (default: <project>/outputs/mem0/store.jsonl)
  BELLA_MEM0_API_KEY            – OpenAI key (fallback: OPENAI_API_KEY)
  BELLA_MEM0_BASE_URL           – Custom base URL (fallback: OPENAI_BASE_URL)
  BELLA_MEM0_LLM_MODEL          – LLM for fact extraction (default: gpt-4o-mini)
  BELLA_MEM0_EMBEDDER_MODEL     – Embedding model (default: text-embedding-3-small)
  BELLA_MEM0_MAX_RESULTS         – Top-k retrieval limit (default: 5)
  BELLA_MEM0_MAX_CHARS_PER_ITEM  – Truncation limit per tool output (default: 800)
  BELLA_MEM0_EXTRACT_FACTS       – true/false; false skips LLM extraction and
                                   embeds raw text directly (default: true)
"""
from __future__ import annotations

import atexit
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from openai import OpenAI

from bella.memory.base import MemoryPlugin
from bella.memory.observation import truncate_tool_output
from bella.memory.registry import register_memory

logger = logging.getLogger(__name__)

_FACT_EXTRACTION_PROMPT = """\
You are a memory extraction assistant. Given a tool interaction record, \
extract the key factual observations as a concise list.

Rules:
- Each fact should be a single, self-contained sentence.
- Focus on results, state changes, and discovered information.
- If the result is an error, note what failed and why.
- Output one fact per line, each starting with "- ".
- If there are no meaningful facts, output exactly: - No notable facts."""


def _find_project_root() -> Path:
    """Walk up from this file to find the directory containing pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


# ── persistent vector store (JSONL + numpy) ──────────────────────────

class _VectorStore:
    """Global, disk-persisted vector store with numpy-accelerated search.

    Data lives in a single JSONL file.  On init the file is loaded into RAM;
    every ``add()`` appends one line and flushes.  Search converts the
    embedding list to a numpy matrix lazily (only when dirty) so that
    repeated adds between searches are cheap.
    """

    def __init__(self, store_path: str) -> None:
        self._texts: List[str] = []
        self._emb_list: List[List[float]] = []
        self._emb_np: np.ndarray | None = None
        self._dirty = True

        self._store_path = store_path
        os.makedirs(os.path.dirname(os.path.abspath(store_path)), exist_ok=True)
        self._load_from_disk()
        self._file = open(store_path, "a", encoding="utf-8")  # noqa: SIM115
        atexit.register(self.close)

    def _load_from_disk(self) -> None:
        if not os.path.exists(self._store_path):
            return
        count = 0
        with open(self._store_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self._texts.append(entry["text"])
                    self._emb_list.append(entry["embedding"])
                    count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
        if count:
            logger.info(
                "mem0 store: loaded %d memories from %s", count, self._store_path
            )

    def close(self) -> None:
        if hasattr(self, "_file") and self._file and not self._file.closed:
            self._file.close()

    def add(self, text: str, embedding: List[float]) -> None:
        self._texts.append(text)
        self._emb_list.append(embedding)
        self._dirty = True

        record = json.dumps(
            {"text": text, "embedding": embedding}, ensure_ascii=False
        )
        self._file.write(record + "\n")
        self._file.flush()

    def _rebuild_np(self) -> None:
        if self._emb_list:
            self._emb_np = np.array(self._emb_list, dtype=np.float32)
        else:
            self._emb_np = None
        self._dirty = False

    def search(self, query_embedding: List[float], limit: int) -> List[str]:
        if not self._texts:
            return []
        if self._dirty:
            self._rebuild_np()
        if self._emb_np is None:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return []

        scores = self._emb_np @ query
        norms = np.linalg.norm(self._emb_np, axis=1)
        scores = scores / (norms * query_norm + 1e-10)

        top_k = min(limit, len(self._texts))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [self._texts[i] for i in top_indices]

    def __len__(self) -> int:
        return len(self._texts)


# ── plugin ───────────────────────────────────────────────────────────

@register_memory("mem0")
class Mem0MemoryPlugin(MemoryPlugin):
    """Global mem0-style memory: LLM fact extraction + vector search + JSONL persistence."""

    def __init__(self) -> None:
        self.max_results: int = int(os.getenv("BELLA_MEM0_MAX_RESULTS", "5"))
        self.max_chars_per_item: int = int(
            os.getenv("BELLA_MEM0_MAX_CHARS_PER_ITEM", "800")
        )
        self._extract_facts: bool = (
            os.getenv("BELLA_MEM0_EXTRACT_FACTS", "true").strip().lower()
            not in ("0", "false", "no")
        )

        api_key = (
            os.getenv("BELLA_MEM0_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or None
        )
        base_url = (
            os.getenv("BELLA_MEM0_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or None
        )
        self._llm_model: str = os.getenv("BELLA_MEM0_LLM_MODEL", "gpt-4o-mini")
        self._embedder_model: str = os.getenv(
            "BELLA_MEM0_EMBEDDER_MODEL", "text-embedding-3-small"
        )

        self._client = OpenAI(api_key=api_key, base_url=base_url)

        store_path = os.getenv("BELLA_MEM0_STORE_PATH") or str(
            _find_project_root() / "outputs" / "mem0" / "store.jsonl"
        )
        self._store = _VectorStore(store_path)
        logger.info(
            "mem0 plugin ready  (store=%s, existing=%d, extract_facts=%s)",
            store_path,
            len(self._store),
            self._extract_facts,
        )

    def _embed(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(
            model=self._embedder_model, input=text
        )
        return resp.data[0].embedding

    def _extract(self, text: str) -> List[str]:
        """Use LLM to distil raw tool output into concise factual sentences."""
        resp = self._client.chat.completions.create(
            model=self._llm_model,
            messages=[
                {"role": "system", "content": _FACT_EXTRACTION_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        content = (resp.choices[0].message.content or "").strip()
        facts = [
            line.lstrip("-").strip()
            for line in content.splitlines()
            if line.strip().startswith("-")
        ]
        if not facts or any("no notable" in f.lower() for f in facts):
            return []
        return facts

    def init_state(self, conversation: Dict[str, Any]) -> None:
        pass

    def on_tool_result(
        self,
        conversation: Dict[str, Any],
        turn_index: int,
        tool_call: str,
        tool_result_raw: str,
    ) -> None:
        truncated = truncate_tool_output(tool_result_raw, self.max_chars_per_item)
        raw_text = f"Called {tool_call}. Result: {truncated}"

        try:
            facts = self._extract(raw_text) if self._extract_facts else []
            if not facts:
                facts = [raw_text]
            for fact in facts:
                embedding = self._embed(fact)
                self._store.add(fact, embedding)
        except Exception as e:
            logger.warning("mem0 add failed (turn %d): %s", turn_index, e)

    def build_prompt_blocks(
        self,
        entry: Dict[str, Any],
        state: Dict[str, Any],
        turn_index: int,
    ) -> Dict[str, str]:
        empty = {"action_history_section": "", "tool_result_memory_section": ""}
        if turn_index == 0 and len(self._store) == 0:
            return empty

        conversation = state["conversation"]
        turn_texts: List[str] = conversation.get("turn_texts", [])
        query = turn_texts[turn_index] if turn_index < len(turn_texts) else ""
        if not query:
            return empty

        try:
            query_embedding = self._embed(query)
            memories = self._store.search(query_embedding, self.max_results)
        except Exception as e:
            logger.warning("mem0 search failed (turn %d): %s", turn_index, e)
            return empty

        if not memories:
            return empty

        lines = [f"- {m}" for m in memories]
        section = (
            "\nRelevant memories from past tool interactions:\n"
            + "\n".join(lines)
            + "\n"
        )
        return {
            "action_history_section": "",
            "tool_result_memory_section": section,
        }
