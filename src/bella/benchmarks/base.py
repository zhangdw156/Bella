"""
Core abstractions for the benchmark framework.

* ``Benchmark``    – describes a benchmark: data loading, adapter creation, evaluation.
* ``InferAdapter`` – per-entry inference lifecycle: build request → parse response → multi-turn loop.

Inspired by evalscope's ``DataAdapter`` / ``BenchmarkMeta`` pattern, but kept
minimal for Bella's scope.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

from bella.infer.types import BellaRequest, BellaResult


# ── Inference adapter ────────────────────────────────────────────────

class InferAdapter(ABC):
    """Abstract per-entry inference adapter.

    Subclasses implement the request/response cycle for a specific benchmark
    or category.  The general runner calls these methods in a standard loop.
    """

    @abstractmethod
    def build_request(self, entry: Dict[str, Any], state: Dict[str, Any]) -> BellaRequest:
        """Build the next LLM request from entry data and current state."""

    @abstractmethod
    def parse_response(
        self,
        entry: Dict[str, Any],
        response: Any,
        state: Dict[str, Any],
    ) -> BellaResult:
        """Parse model response, update state, return intermediate result."""

    def on_run_start(self, session_id: str) -> None:
        """Called once before the inference loop.

        *session_id* identifies this run (e.g. ``"bfcl/multi_turn_base"``).
        Adapters with long-term memory should forward to ``memory_plugin.open()``.
        """

    def on_run_end(self) -> None:
        """Called once after the inference loop (including on error).

        Adapters with long-term memory should forward to ``memory_plugin.close()``.
        """

    def init_state(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Initialise per-entry state (default: empty dict)."""
        return {}

    def has_next_turn(self, entry: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """Whether the entry still has remaining turns (default: False → single-turn)."""
        return False

    def finalize_result(
        self,
        entry: Dict[str, Any],
        state: Dict[str, Any],
        last_result: BellaResult,
    ) -> BellaResult:
        """Post-processing hook to consolidate multi-turn results."""
        return last_result


# ── Benchmark ────────────────────────────────────────────────────────

class Benchmark(ABC):
    """Abstract benchmark definition.

    A ``Benchmark`` knows how to:
    1. Load its dataset,
    2. Create an inference adapter for a category,
    3. Compute result file paths,
    4. Run evaluation / scoring.

    Each benchmark self-registers via ``@register_benchmark`` so the
    general runner can instantiate it by name.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique benchmark identifier (e.g. ``"bfcl"``, ``"locomo"``)."""

    @abstractmethod
    def list_categories(self) -> List[str]:
        """Return available category / subset names."""

    @abstractmethod
    def load_dataset(self, category: str, limit: int = 0) -> List[Dict[str, Any]]:
        """Load dataset entries for *category*.  *limit* ≤ 0 means all."""

    @abstractmethod
    def create_adapter(self, category: str) -> InferAdapter:
        """Return an adapter instance for *category*."""

    @abstractmethod
    def result_file(self, category: str) -> Path:
        """Return the absolute path to the result JSONL for *category*."""

    @abstractmethod
    def evaluate(self, category: str, **kwargs: Any) -> None:
        """Run evaluation / scoring for *category*."""
