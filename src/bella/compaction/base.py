from __future__ import annotations

from typing import Protocol

from bella.model.base import Model
from bella.memory.base import Memory


class ContextCompactor(Protocol):
    def compact(self, memory: Memory, model: Model) -> Memory:
        """Compact memory to fit within token budget.

        Invariants:
        - system_prompt is always preserved.
        - Returns a new Memory instance (does not mutate input).
        """
        ...


class NoopCompactor:
    def compact(self, memory: Memory, model: Model) -> Memory:
        return memory
