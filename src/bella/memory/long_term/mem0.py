"""
Self-contained mem0-style memory plugin with session lifecycle and thread safety.

Core interface (per Mem0 paper):
  add(content)    – LLM fact extraction → embed → store in vector DB
  search(query)   – embed query → cosine similarity → return top-k

The legacy BFCL hooks (on_tool_result / build_prompt_blocks) are inherited
from MemoryPlugin base class which delegates to add() / search().

Lifecycle:
  open(session_id)  → fresh isolated store
  inference loop    → concurrent add/search (RLock-protected)
  close()           → destroy store + delete files
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from openai import OpenAI

from bella.memory.base import MemoryPlugin
from bella.memory.registry import register_memory

logger = logging.getLogger(__name__)

_FACT_EXTRACTION_PROMPT = """\
You are a memory extraction assistant. Given a piece of information, \
extract the key factual observations as a concise list.

Rules:
- Each fact should be a single, self-contained sentence.
- Focus on results, state changes, and discovered information.
- If the result is an error, note what failed and why.
- Output one fact per line, each starting with "- ".
- If there are no meaningful facts, output exactly: - No notable facts."""


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


# ── thread-safe persistent vector store ──────────────────────────────

class _VectorStore:
    """Thread-safe, disk-persisted vector store with numpy-accelerated search."""

    def __init__(self, store_path: str) -> None:
        self._lock = threading.RLock()
        self._texts: List[str] = []
        self._emb_list: List[List[float]] = []
        self._emb_np: np.ndarray | None = None
        self._dirty = True
        self._store_path = store_path

        os.makedirs(os.path.dirname(os.path.abspath(store_path)), exist_ok=True)
        self._load_from_disk()
        self._file = open(store_path, "a", encoding="utf-8")  # noqa: SIM115

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
            logger.info("mem0 store: loaded %d memories from %s", count, self._store_path)

    def add(self, text: str, embedding: List[float]) -> None:
        with self._lock:
            self._texts.append(text)
            self._emb_list.append(embedding)
            self._dirty = True
            record = json.dumps({"text": text, "embedding": embedding}, ensure_ascii=False)
            self._file.write(record + "\n")
            self._file.flush()

    def _rebuild_np(self) -> None:
        if self._emb_list:
            self._emb_np = np.array(self._emb_list, dtype=np.float32)
        else:
            self._emb_np = None
        self._dirty = False

    def search(self, query_embedding: List[float], limit: int) -> List[str]:
        with self._lock:
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

    def close(self) -> None:
        with self._lock:
            if hasattr(self, "_file") and self._file and not self._file.closed:
                self._file.close()

    def destroy(self) -> None:
        self.close()
        store_dir = os.path.dirname(os.path.abspath(self._store_path))
        if os.path.isdir(store_dir):
            shutil.rmtree(store_dir, ignore_errors=True)

    def __len__(self) -> int:
        with self._lock:
            return len(self._texts)


# ── plugin ───────────────────────────────────────────────────────────

@register_memory("mem0")
class Mem0MemoryPlugin(MemoryPlugin):
    """Mem0-style memory: LLM fact extraction + vector search.

    Implements the general ``add`` / ``search`` interface.  The BFCL-specific
    ``on_tool_result`` / ``build_prompt_blocks`` hooks are inherited from the
    base class and delegate to ``add`` / ``search`` automatically.
    """

    def __init__(self) -> None:
        self.max_results: int = int(os.getenv("BELLA_MEM0_MAX_RESULTS", "5"))
        self._extract_facts: bool = (
            os.getenv("BELLA_MEM0_EXTRACT_FACTS", "true").strip().lower()
            not in ("0", "false", "no")
        )

        api_key = os.getenv("BELLA_MEM0_API_KEY") or os.getenv("OPENAI_API_KEY") or None
        base_url = os.getenv("BELLA_MEM0_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None
        self._llm_model: str = os.getenv("BELLA_MEM0_LLM_MODEL", "gpt-4o-mini")
        self._embedder_model: str = os.getenv("BELLA_MEM0_EMBEDDER_MODEL", "text-embedding-3-small")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

        self._store_dir = os.getenv("BELLA_MEM0_STORE_DIR") or str(
            _find_project_root() / "outputs" / "mem0"
        )
        self._store: _VectorStore | None = None
        self._session_id: str | None = None

    # ── lifecycle ────────────────────────────────────────────────────

    def open(self, session_id: str) -> None:
        if self._store is not None:
            self.close()
        self._session_id = session_id
        safe_name = session_id.replace("/", "_").replace("\\", "_")
        store_path = os.path.join(self._store_dir, safe_name, "store.jsonl")
        self._store = _VectorStore(store_path)
        logger.info("mem0 opened  session=%r  store=%s", session_id, store_path)

    def close(self) -> None:
        if self._store is not None:
            count = len(self._store)
            self._store.destroy()
            self._store = None
            logger.info("mem0 closed  session=%r  memories_cleared=%d", self._session_id, count)
            self._session_id = None

    # ── internal helpers ─────────────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        resp = self._client.embeddings.create(model=self._embedder_model, input=text)
        return resp.data[0].embedding

    def _extract(self, text: str) -> List[str]:
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

    # ── core interface ───────────────────────────────────────────────

    def add(self, content: str, metadata: Dict[str, Any] | None = None) -> None:
        if self._store is None:
            return
        try:
            facts = self._extract(content) if self._extract_facts else []
            if not facts:
                facts = [content]
            for fact in facts:
                embedding = self._embed(fact)
                self._store.add(fact, embedding)
        except Exception as e:
            logger.warning("mem0 add failed: %s", e)

    def search(self, query: str, limit: int = 5) -> List[str]:
        if self._store is None or len(self._store) == 0:
            return []
        try:
            query_embedding = self._embed(query)
            return self._store.search(query_embedding, limit)
        except Exception as e:
            logger.warning("mem0 search failed: %s", e)
            return []
