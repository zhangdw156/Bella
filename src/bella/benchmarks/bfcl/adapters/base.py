"""
BFCL adapter base class and per-category registry.

``BFCLAdapter`` extends the general ``InferAdapter`` with BFCL-specific
``result_group`` / ``result_filename`` methods used by ``BFCLBenchmark``
to compute result paths.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict

from bella.benchmarks.base import InferAdapter


class BFCLAdapter(InferAdapter):
    """Base class for BFCL category adapters."""

    @abstractmethod
    def result_group(self, category: str) -> str:
        """High-level group (e.g. ``'non_live'``, ``'multi_turn'``)."""

    @abstractmethod
    def result_filename(self, category: str) -> str:
        """JSONL result filename for the given category."""


# ── per-category registry ────────────────────────────────────────────

_ADAPTERS: Dict[str, type[BFCLAdapter]] = {}


def register_adapter(category: str):
    """Decorator that registers a ``BFCLAdapter`` under *category*."""
    def _wrap(cls: type[BFCLAdapter]) -> type[BFCLAdapter]:
        _ADAPTERS[category] = cls
        return cls
    return _wrap


def get_adapter(category: str) -> BFCLAdapter:
    """Instantiate the registered adapter for *category*."""
    cls = _ADAPTERS.get(category)
    if cls is None:
        available = ", ".join(sorted(_ADAPTERS.keys())) or "(none)"
        raise ValueError(
            f"No BFCL adapter registered for category {category!r}. "
            f"Available: {available}"
        )
    return cls()
