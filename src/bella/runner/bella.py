"""BellaRunner: unified evaluation runner wrapping CaseRunner + ReplayRunner."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bella.agent.react import ReactAgent
from bella.agent.user import UserAgent
from bella.model.base import Model
from bella.runner.case import CaseRunner, CaseResult
from bella.runner.replay import ReplayRunner, ReplayResult
from bella.types import Case, Message


def _serialize_case_result(result: CaseResult) -> dict:
    return {
        "case_id": result.case_id,
        "env_name": result.env_name,
        "category": result.category,
        "interaction_mode": result.interaction_mode,
        "react_model_config": result.react_model_config,
        "user_model_config": result.user_model_config,
        "ended_normally": result.ended_normally,
        "tool_calls": result.tool_calls,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "tool_calls": (
                    [{"name": tc.name, "arguments": tc.arguments, "id": tc.id} for tc in m.tool_calls]
                    if m.tool_calls
                    else None
                ),
                "tool_call_id": m.tool_call_id,
                "reasoning_content": m.reasoning_content,
            }
            for m in result.messages
        ],
        "token_usage": (
            {"input_tokens": result.token_usage.input_tokens, "output_tokens": result.token_usage.output_tokens}
            if result.token_usage
            else None
        ),
        "timing": {"duration": result.timing.duration},
    }


def _serialize_replay_result(case_id: str, replay: ReplayResult) -> dict:
    return {
        "case_id": case_id,
        "total_calls": replay.total_calls,
        "matched": replay.matched,
        "mismatched": replay.mismatched,
        "token_substitutions": replay.token_substitutions,
        "verify": [
            {"sql": vr.sql, "expected": vr.expected, "actual": vr.actual, "passed": vr.passed}
            for vr in replay.verify_results
        ],
        "passed": replay.passed,
    }


@dataclass
class BellaResult:
    case_id: str
    trial: int
    case_result: CaseResult | None
    replay_result: ReplayResult | None
    passed: bool
    error: str | None
    duration: float

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "case_id": self.case_id,
            "trial": self.trial,
            "passed": self.passed,
            "error": self.error,
            "duration": self.duration,
        }
        if self.case_result:
            d["case_result"] = _serialize_case_result(self.case_result)
        if self.replay_result:
            d["replay_result"] = _serialize_replay_result(self.case_id, self.replay_result)
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


@dataclass
class CaseEvalResult:
    n_trials: int
    n_pass: int
    pass_rate: float
    pass_at_k: bool
    pass_pow_k: bool
    trials: list[dict]


@dataclass
class EvalResult:
    react_model_id: str
    user_model_id: str
    n_cases: int
    n: int
    total_pass: int
    total_trials: int
    pass_at_1: float
    pass_at_k: float
    pass_pow_k: float
    per_case: dict[str, CaseEvalResult]
    eval_duration: float
    results: list[BellaResult] = field(default_factory=list, repr=False)

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE  ({self.eval_duration:.1f}s)")
        print(f"{'='*60}")
        print(f"  React model: {self.react_model_id}")
        print(f"  User model:  {self.user_model_id}")
        print(f"  Cases:       {self.n_cases}")
        print(f"  Trials:      {self.total_trials}")
        n_pass_at_k = sum(1 for v in self.per_case.values() if v.pass_at_k)
        n_pass_pow_k = sum(1 for v in self.per_case.values() if v.pass_pow_k)
        print(f"  pass@1:      {self.pass_at_1:.3f}  ({self.total_pass}/{self.total_trials})")
        if self.n > 1:
            print(f"  pass@{self.n}:      {self.pass_at_k:.3f}  ({n_pass_at_k}/{self.n_cases})")
            print(f"  pass^{self.n}:      {self.pass_pow_k:.3f}  ({n_pass_pow_k}/{self.n_cases})")
        print()
        print("Per-case breakdown:")
        for case_id, info in sorted(self.per_case.items()):
            trials_str = "".join("P" if t["passed"] else "F" for t in info.trials)
            print(f"  {case_id}  [{trials_str}]  {info.n_pass}/{info.n_trials}")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "react_model_id": self.react_model_id,
            "user_model_id": self.user_model_id,
            "n_cases": self.n_cases,
            "n": self.n,
            "total_pass": self.total_pass,
            "total_trials": self.total_trials,
            "pass_at_1": self.pass_at_1,
            f"pass_at_{self.n}": self.pass_at_k,
            f"pass_pow_{self.n}": self.pass_pow_k,
            "eval_duration": self.eval_duration,
            "per_case": {
                cid: {
                    "n_trials": v.n_trials,
                    "n_pass": v.n_pass,
                    "pass_rate": v.pass_rate,
                    f"pass_at_{self.n}": v.pass_at_k,
                    f"pass_pow_{self.n}": v.pass_pow_k,
                    "trials": v.trials,
                }
                for cid, v in sorted(self.per_case.items())
            },
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def _clone_model(model: Model) -> Model:
    return type(model)(
        model_id=model.model_id,
        base_url=model.base_url,
        api_key=model.api_key,
        temperature=model.temperature,
        max_context_tokens=model.max_context_tokens,
    )


class BellaRunner:
    def __init__(
        self,
        react_model: Model,
        user_model: Model,
        cases: list[Case],
        n: int = 1,
        concurrency: int = 1,
        environments_dir: Path = Path("environments"),
        category_prompts: dict | None = None,
        max_turns: int = 30,
        max_llm_calls_per_turn: int = 12,
        output_dir: Path | None = None,
    ):
        self._react_model = react_model
        self._user_model = user_model
        self.cases = cases
        self.n = n
        self.concurrency = concurrency
        self.environments_dir = environments_dir
        self.max_turns = max_turns
        self.max_llm_calls_per_turn = max_llm_calls_per_turn
        self.output_dir = output_dir

        if category_prompts is None:
            with open("category_prompts.json") as f:
                category_prompts = json.load(f)
        self.category_prompts = category_prompts

    def _run_single(self, case: Case, trial: int) -> BellaResult:
        react_model = _clone_model(self._react_model)
        user_model = _clone_model(self._user_model)

        react_agent = ReactAgent(model=react_model, max_llm_calls_per_turn=self.max_llm_calls_per_turn)
        user_agent = UserAgent(model=user_model, max_turns=self.max_turns)

        case_runner = CaseRunner(
            react_agent=react_agent,
            user_agent=user_agent,
            environments_dir=self.environments_dir,
            category_prompts=self.category_prompts,
            max_turns=self.max_turns,
        )
        replay_runner = ReplayRunner(environments_dir=self.environments_dir)

        t0 = time.time()
        case_result = None
        replay_result = None
        passed = False
        error = None

        try:
            case_result = case_runner.run(case)
            replay_result = replay_runner.run(case, case_result.tool_calls)
            passed = replay_result.passed
        except Exception as e:
            error = str(e)

        duration = time.time() - t0

        bella_result = BellaResult(
            case_id=case.case_id,
            trial=trial,
            case_result=case_result,
            replay_result=replay_result,
            passed=passed,
            error=error,
            duration=duration,
        )

        if self.output_dir:
            bella_result.save(self.output_dir / "runs" / f"{case.case_id}_trial{trial}.json")

        status = "PASS" if passed else ("ERROR" if error else "FAIL")
        tool_count = len(case_result.tool_calls) if case_result else 0
        print(f"  [{status}] {case.case_id}/trial{trial}  ({duration:.1f}s, {tool_count} tools)")

        return bella_result

    def _compute_metrics(self, results: list[BellaResult], eval_duration: float) -> EvalResult:
        case_trials: dict[str, list[BellaResult]] = {}
        for r in results:
            case_trials.setdefault(r.case_id, []).append(r)
        for trials in case_trials.values():
            trials.sort(key=lambda t: t.trial)

        per_case: dict[str, CaseEvalResult] = {}
        for case_id, trials in sorted(case_trials.items()):
            n_t = len(trials)
            n_pass = sum(1 for t in trials if t.passed)
            per_case[case_id] = CaseEvalResult(
                n_trials=n_t,
                n_pass=n_pass,
                pass_rate=n_pass / n_t,
                pass_at_k=n_pass >= 1,
                pass_pow_k=n_pass == n_t,
                trials=[
                    {"trial": t.trial, "passed": t.passed, "duration": t.duration, "error": t.error}
                    for t in trials
                ],
            )

        total_pass = sum(v.n_pass for v in per_case.values())
        total_trials = sum(v.n_trials for v in per_case.values())
        n_cases = len(per_case)

        return EvalResult(
            react_model_id=self._react_model.model_id,
            user_model_id=self._user_model.model_id,
            n_cases=n_cases,
            n=self.n,
            total_pass=total_pass,
            total_trials=total_trials,
            pass_at_1=total_pass / total_trials if total_trials else 0,
            pass_at_k=sum(1 for v in per_case.values() if v.pass_at_k) / n_cases if n_cases else 0,
            pass_pow_k=sum(1 for v in per_case.values() if v.pass_pow_k) / n_cases if n_cases else 0,
            per_case=per_case,
            eval_duration=eval_duration,
            results=results,
        )

    def run(self) -> EvalResult:
        print(f"React model: {self._react_model.model_id}")
        print(f"User model:  {self._user_model.model_id}")
        print(f"Cases:       {len(self.cases)}")
        print(f"N:           {self.n}")
        print(f"Concurrency: {self.concurrency}")
        if self.output_dir:
            print(f"Output:      {self.output_dir}")
        print()

        eval_start = time.time()
        all_results: list[BellaResult] = []

        if self.concurrency <= 1:
            for case in self.cases:
                for trial in range(self.n):
                    all_results.append(self._run_single(case, trial))
        else:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                futures = {}
                for case in self.cases:
                    for trial in range(self.n):
                        fut = pool.submit(self._run_single, case, trial)
                        futures[fut] = (case.case_id, trial)

                for fut in as_completed(futures):
                    case_id, trial = futures[fut]
                    try:
                        all_results.append(fut.result())
                    except Exception as e:
                        print(f"  [FATAL] {case_id}/trial{trial}: {e}", file=sys.stderr)
                        all_results.append(BellaResult(
                            case_id=case_id,
                            trial=trial,
                            case_result=None,
                            replay_result=None,
                            passed=False,
                            error=str(e),
                            duration=0,
                        ))

        eval_duration = time.time() - eval_start
        eval_result = self._compute_metrics(all_results, eval_duration)

        if self.output_dir:
            eval_result.save(self.output_dir / "summary.json")

        return eval_result
