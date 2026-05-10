"""Replay runner: replays tool call chain on fresh DB and verifies results."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bella.types import Case


@dataclass
class VerifyResult:
    sql: str
    expected: list[list]
    actual: list[list]
    passed: bool


@dataclass
class ReplayResult:
    total_calls: int
    matched: int
    mismatched: int
    token_substitutions: int
    verify_results: list[VerifyResult]
    passed: bool


def _load_backend(backend_path: Path, db_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("_replay_backend", backend_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EnvironmentBackend(db_path=db_path)


def _extract_token(result: Any) -> str | None:
    """Extract token-like values from a tool result for substitution."""
    if not isinstance(result, dict):
        return None
    for key in ("access_token", "session_id", "token"):
        if key in result and isinstance(result[key], str):
            return result[key]
    data = result.get("data")
    if isinstance(data, dict):
        for key in ("access_token", "session_id", "token"):
            if key in data and isinstance(data[key], str):
                return data[key]
    return None


def _substitute_tokens(args: dict[str, Any], token_map: dict[str, str]) -> dict[str, Any]:
    """Deep-copy args and replace any old token values with new ones."""
    if not token_map:
        return args
    return _substitute_recursive(copy.deepcopy(args), token_map)


def _substitute_recursive(obj: Any, token_map: dict[str, str]) -> Any:
    if isinstance(obj, str):
        return token_map.get(obj, obj)
    if isinstance(obj, dict):
        return {k: _substitute_recursive(v, token_map) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_recursive(item, token_map) for item in obj]
    return obj


def _compare_results(expected: list[list], actual: list[list], order_matters: bool) -> bool:
    if order_matters:
        return expected == actual
    return sorted([sorted(str(x) for x in row) for row in expected]) == \
           sorted([sorted(str(x) for x in row) for row in actual])


class ReplayRunner:
    def __init__(self, environments_dir: Path = Path("environments")):
        self.environments_dir = environments_dir

    def run(self, case: Case, tool_calls: list[dict[str, Any]]) -> ReplayResult:
        """Replay tool calls on a fresh DB and verify results.

        Args:
            case: The case definition (needs env_name, world_setup, verify).
            tool_calls: Tool call chain from CaseResult.tool_calls.

        Returns:
            ReplayResult with replay stats and verification results.
        """
        env_name = case.env_name
        world_setup = case.world_setup
        verify = [{"sql": v.sql, "expected": v.expected, "order_matters": v.order_matters} for v in case.verify]

        env_dir = self.environments_dir / env_name

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"bella_replay_{env_name}_"))
        world_db = env_dir / "world" / "world.db"
        replay_db = tmp_dir / "replay.db"
        shutil.copy2(world_db, replay_db)

        try:
            # Execute world_setup SQL
            if world_setup:
                with sqlite3.connect(str(replay_db)) as conn:
                    for sql in world_setup:
                        conn.execute(sql)
                    conn.commit()

            # Load backend
            backend_path = env_dir / "runtime" / "backend.py"
            backend = _load_backend(backend_path, replay_db)

            # Replay tool calls with token substitution
            token_map: dict[str, str] = {}
            matched = 0
            mismatched = 0
            substitutions = 0

            for tc in tool_calls:
                patched_args = _substitute_tokens(tc["arguments"], token_map)
                if patched_args != tc["arguments"]:
                    substitutions += 1

                replay_result = backend.call(tc["name"], patched_args)

                orig_token = _extract_token(tc.get("result"))
                replay_token = _extract_token(replay_result)
                if orig_token and replay_token and orig_token != replay_token:
                    token_map[orig_token] = replay_token

                original_json = json.dumps(tc.get("result"), sort_keys=True, ensure_ascii=False)
                replay_json = json.dumps(replay_result, sort_keys=True, ensure_ascii=False)
                if original_json == replay_json:
                    matched += 1
                else:
                    mismatched += 1

            # Verify SQL
            verify_results: list[VerifyResult] = []
            with sqlite3.connect(str(replay_db)) as conn:
                for v in verify:
                    sql = v["sql"]
                    expected = v["expected"]
                    order_matters = v.get("order_matters", False)

                    cursor = conn.execute(sql)
                    actual = [list(row) for row in cursor.fetchall()]

                    passed = _compare_results(expected, actual, order_matters)
                    verify_results.append(VerifyResult(
                        sql=sql,
                        expected=expected,
                        actual=actual,
                        passed=passed,
                    ))

            all_passed = all(vr.passed for vr in verify_results)

            return ReplayResult(
                total_calls=len(tool_calls),
                matched=matched,
                mismatched=mismatched,
                token_substitutions=substitutions,
                verify_results=verify_results,
                passed=all_passed,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
