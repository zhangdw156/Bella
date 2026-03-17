from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from bella.infer.types import BellaRequest, BellaResult


class BFCLAdapter(ABC):
    @abstractmethod
    def build_request(self, entry: Dict[str, Any], state: Dict[str, Any]) -> BellaRequest:
        ...

    @abstractmethod
    def parse_response(
        self,
        entry: Dict[str, Any],
        response: Any,
        state: Dict[str, Any],
    ) -> BellaResult:
        ...

    @abstractmethod
    def result_group(self, category: str) -> str:
        """
        Return high-level group name (e.g. 'non_live', 'live', 'multi_turn', 'agentic').
        """
        ...

    @abstractmethod
    def result_filename(self, category: str) -> str:
        """
        Return the JSONL result filename for the given category.
        """
        ...

    # ---- Optional multi-turn hooks ----

    def init_state(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize per-entry state. Single-turn adapters can rely on the
        default empty state.
        """
        return {}

    def has_next_turn(self, entry: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """
        Indicate whether the current entry still has remaining turns.

        Default False keeps existing single-turn behavior unchanged.
        """
        return False

    def finalize_result(
        self,
        entry: Dict[str, Any],
        state: Dict[str, Any],
        last_result: BellaResult,
    ) -> BellaResult:
        """
        Final post-processing hook to consolidate multi-turn results into a
        single BellaResult. Single-turn adapters can keep the last_result.
        """
        return last_result


_ADAPTERS: Dict[str, type[BFCLAdapter]] = {}


def register_adapter(category: str):
    def _wrap(cls: type[BFCLAdapter]):
        _ADAPTERS[category] = cls
        return cls

    return _wrap


def get_adapter(category: str) -> BFCLAdapter:
    cls = _ADAPTERS.get(category)
    if cls is None:
        raise ValueError(f"No BFCL adapter registered for category '{category}'")
    return cls()

