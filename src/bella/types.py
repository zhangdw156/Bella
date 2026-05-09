from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Timing:
    start_time: float
    end_time: float | None = None

    @property
    def duration(self) -> float | None:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str


@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    reasoning_content: str | None = None
    token_usage: TokenUsage | None = None


@runtime_checkable
class Backend(Protocol):
    def call(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class UserAgentConfig:
    role: str
    personality: str
    knowledge_boundary: str


@dataclass
class VerifySpec:
    sql: str
    expected: list[list]
    order_matters: bool = False


# TODO: support case variants (inspired by BFCL v4):
#   - miss_func: hide a tool in certain turns, test if agent recognizes the gap
#   - miss_param: make user instructions vague in one turn, clarify in the next
#   This lets one base case generate multiple test instances testing different abilities.

@dataclass
class Case:
    case_id: str
    env_name: str
    category: str
    source: str
    tags: list[str]
    interaction_mode: Literal["fixed", "dynamic"]
    demand: str | None = None
    user_demands: list[str] | None = None
    world_setup: list[str] = field(default_factory=list)
    user_agent_config: UserAgentConfig | None = None
    verify: list[VerifySpec] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: Path) -> Case:
        with open(path) as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def load_dir(cls, cases_dir: Path, pattern: str = "*.json") -> list[Case]:
        cases = []
        for p in sorted(cases_dir.glob(pattern)):
            cases.append(cls.from_json(p))
        return cases

    @classmethod
    def _from_dict(cls, data: dict) -> Case:
        uac = data.get("user_agent_config")
        if uac is not None:
            uac = UserAgentConfig(**uac)
        verify = [VerifySpec(**v) for v in data.get("verify", [])]
        return cls(
            case_id=data["case_id"],
            env_name=data["env_name"],
            category=data["category"],
            source=data["source"],
            tags=data.get("tags", []),
            interaction_mode=data["interaction_mode"],
            demand=data.get("demand"),
            user_demands=data.get("user_demands"),
            world_setup=data.get("world_setup", []),
            user_agent_config=uac,
            verify=verify,
        )
