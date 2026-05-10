from __future__ import annotations

import json
import shutil
import importlib.util
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bella.types import Message, ToolCall, TokenUsage, Timing, Case
from bella.model.base import Model
from bella.agent.react import ReactAgent, TurnResult
from bella.agent.user import UserAgent, UserTurnResult


@dataclass
class CaseResult:
    case_id: str
    env_name: str
    category: str
    interaction_mode: str
    react_model_config: dict
    user_model_config: dict | None
    messages: list[Message]
    tool_calls: list[dict[str, Any]]
    token_usage: TokenUsage | None
    timing: Timing
    ended_normally: bool


class CaseRunner:
    def __init__(
        self,
        react_agent: ReactAgent,
        user_agent: UserAgent | None = None,
        environments_dir: Path = Path("environments"),
        category_prompts: dict[str, str | None] | None = None,
        max_turns: int = 30,
    ):
        self.react_agent = react_agent
        self.user_agent = user_agent
        self.environments_dir = environments_dir
        self.category_prompts = category_prompts or {}
        self.max_turns = max_turns

    def _load_environment(self, env_name: str, world_setup: list[str]) -> tuple[Any, list[dict], Path]:
        """Load backend and tools for the given environment.

        Returns (backend, tools, tmp_dir) — caller must clean up tmp_dir.
        """
        env_dir = self.environments_dir / env_name

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"bella_{env_name}_"))
        world_db = env_dir / "world" / "world.db"
        session_db = tmp_dir / "session.db"
        shutil.copy2(world_db, session_db)

        # Execute world_setup SQL
        if world_setup:
            import sqlite3
            with sqlite3.connect(str(session_db)) as conn:
                for sql in world_setup:
                    conn.execute(sql)
                conn.commit()

        # Load backend
        backend_path = env_dir / "runtime" / "backend.py"
        spec = importlib.util.spec_from_file_location("backend", backend_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        backend = module.EnvironmentBackend(db_path=session_db)

        # Load tools
        tools_path = env_dir / "contract" / "tools.jsonl"
        tools = []
        with open(tools_path) as f:
            for line in f:
                tools.append(json.loads(line))

        return backend, tools, tmp_dir

    def _build_system_prompt(self, category: str) -> str:
        return self.category_prompts.get(category)

    def _collect_tool_calls(self) -> list[dict[str, Any]]:
        """Extract the tool call chain from react agent memory."""
        calls = []
        for turn in self.react_agent.memory.turns:
            for assistant_step, tool_results in turn.react_steps:
                for tc, tr in zip(assistant_step.tool_calls, tool_results):
                    calls.append({
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "id": tc.id,
                        "result": tr.result,
                    })
        return calls

    def _collect_messages(self) -> list[Message]:
        """Extract full message history from react agent memory."""
        return self.react_agent.memory.to_messages(
            current_turn_index=len(self.react_agent.memory.turns) - 1
        )

    def run(self, case: Case) -> CaseResult:
        """Run a single case and return the result."""
        case_id = case.case_id
        env_name = case.env_name
        category = case.category
        interaction_mode = case.interaction_mode
        world_setup = case.world_setup

        start_time = time.time()

        backend, tools, tmp_dir = self._load_environment(env_name, world_setup)
        try:
            category_prompt = self._build_system_prompt(category)
            self.react_agent.init_memory(interaction_mode=interaction_mode, extra_system_prompt=category_prompt)

            total_token_usage: TokenUsage | None = None
            ended_normally = False

            if interaction_mode == "fixed":
                user_demands = case.user_demands or []
                for user_msg in user_demands:
                    result = self.react_agent.run_turn(user_msg, tools, backend)
                    if result.token_usage:
                        if total_token_usage is None:
                            total_token_usage = result.token_usage
                        else:
                            total_token_usage = TokenUsage(
                                input_tokens=total_token_usage.input_tokens + result.token_usage.input_tokens,
                                output_tokens=total_token_usage.output_tokens + result.token_usage.output_tokens,
                            )
                ended_normally = True

            elif interaction_mode == "dynamic":
                assert self.user_agent is not None
                demand = case.demand
                user_agent_config = {
                    "role": case.user_agent_config.role,
                    "personality": case.user_agent_config.personality,
                    "knowledge_boundary": case.user_agent_config.knowledge_boundary,
                } if case.user_agent_config else {}

                user_result = self.user_agent.start(demand, user_agent_config)
                turn_count = 0

                while not user_result.is_done and turn_count < self.max_turns:
                    result = self.react_agent.run_turn(user_result.message, tools, backend)
                    if result.token_usage:
                        if total_token_usage is None:
                            total_token_usage = result.token_usage
                        else:
                            total_token_usage = TokenUsage(
                                input_tokens=total_token_usage.input_tokens + result.token_usage.input_tokens,
                                output_tokens=total_token_usage.output_tokens + result.token_usage.output_tokens,
                            )
                    user_result = self.user_agent.respond(result.assistant_message)
                    turn_count += 1

                ended_normally = user_result.is_done

            end_time = time.time()

            return CaseResult(
                case_id=case_id,
                env_name=env_name,
                category=category,
                interaction_mode=interaction_mode,
                react_model_config=self.react_agent.model.to_config(),
                user_model_config=self.user_agent.model.to_config() if self.user_agent else None,
                messages=self._collect_messages(),
                tool_calls=self._collect_tool_calls(),
                token_usage=total_token_usage,
                timing=Timing(start_time=start_time, end_time=end_time),
                ended_normally=ended_normally,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
