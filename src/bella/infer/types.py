from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BellaRequest:
    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: str = "auto"
    temperature: float = 0.0


@dataclass
class BellaResult:
    id: str
    result: List[Dict[str, str]]
    input_token_count: int = 0
    output_token_count: int = 0
    latency: float = 0.0
    extra: Optional[Dict[str, Any]] = None

