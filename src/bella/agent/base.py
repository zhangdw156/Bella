from __future__ import annotations

from abc import ABC

from bella.model.base import Model
from bella.compaction.base import ContextCompactor


class Agent(ABC):
    def __init__(
        self,
        model: Model,
        compactor: ContextCompactor | None = None,
    ):
        self.model = model
        self._compactor = compactor

    @property
    def compactor(self) -> ContextCompactor:
        if self._compactor is not None:
            return self._compactor
        return self.default_compactor()

    def default_compactor(self) -> ContextCompactor:
        from bella.compaction.base import NoopCompactor
        return NoopCompactor()
