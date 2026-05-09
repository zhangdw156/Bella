#!/usr/bin/env python3
"""Concurrent evaluation of all cases × 4 trials. Computes pass@4 and pass^4."""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from bella.agent.react import ReactAgent
from bella.agent.user import UserAgent
from bella.model.openai_chat import OpenAIChatModel
from bella.runner.case import CaseRunner, CaseResult
from bella.runner.replay import ReplayRunner

load_dotenv()


def serialize_result(result: CaseResult) -> dict:
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


def run_single_trial(
    case: dict,
    trial: int,
    model_kwargs: dict,
    category_prompts: dict,
    environments_dir: Path,
    output_dir: Path,
) -> dict:
    """Run one case × one trial. Returns a per-trial summary dict."""
    case_id = case["case_id"]
    tag = f"{case_id}/trial{trial}"

    react_model = OpenAIChatModel(**model_kwargs)
    user_model = OpenAIChatModel(**model_kwargs)
    react_agent = ReactAgent(model=react_model, max_llm_calls_per_turn=12)
    user_agent = UserAgent(model=user_model, max_turns=30)

    runner = CaseRunner(
        react_agent=react_agent,
        user_agent=user_agent,
        environments_dir=environments_dir,
        category_prompts=category_prompts,
        max_turns=30,
    )

    replay_runner = ReplayRunner(environments_dir=environments_dir)

    t0 = time.time()
    error = None
    passed = False
    verify_details = []
    run_output = None
    replay_output = None

    try:
        result = runner.run(case)
        run_output = serialize_result(result)

        replay = replay_runner.run(case, result.tool_calls)
        passed = replay.passed
        verify_details = [
            {"sql": vr.sql, "expected": vr.expected, "actual": vr.actual, "passed": vr.passed}
            for vr in replay.verify_results
        ]
        replay_output = {
            "case_id": case_id,
            "total_calls": replay.total_calls,
            "matched": replay.matched,
            "mismatched": replay.mismatched,
            "token_substitutions": replay.token_substitutions,
            "verify": verify_details,
            "passed": replay.passed,
        }

        tool_call_count = len(result.tool_calls)
        ended_normally = result.ended_normally
        token_usage = (
            {"input_tokens": result.token_usage.input_tokens, "output_tokens": result.token_usage.output_tokens}
            if result.token_usage
            else None
        )
    except Exception as e:
        error = str(e)
        tool_call_count = 0
        ended_normally = False
        token_usage = None

    duration = time.time() - t0

    if run_output:
        p = output_dir / "runs" / f"{case_id}_trial{trial}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(run_output, f, indent=2, ensure_ascii=False)
    if replay_output:
        p = output_dir / "runs" / f"{case_id}_trial{trial}_replay.json"
        with open(p, "w") as f:
            json.dump(replay_output, f, indent=2, ensure_ascii=False)

    status = "PASS" if passed else ("ERROR" if error else "FAIL")
    print(f"  [{status}] {tag}  ({duration:.1f}s, {tool_call_count} tools)")

    return {
        "case_id": case_id,
        "trial": trial,
        "passed": passed,
        "error": error,
        "duration": duration,
        "tool_calls": tool_call_count,
        "ended_normally": ended_normally,
        "token_usage": token_usage,
        "verify": verify_details,
    }


def compute_metrics(case_trials: dict[str, list[dict]]) -> dict:
    """Compute per-case and aggregate metrics from trial results."""
    per_case = {}
    for case_id, trials in sorted(case_trials.items()):
        n = len(trials)
        passes = sum(1 for t in trials if t["passed"])
        per_case[case_id] = {
            "n_trials": n,
            "n_pass": passes,
            "pass_rate": passes / n,
            "pass_at_4": int(passes >= 1),
            "pass_pow_4": int(passes == n),
            "trials": [
                {"trial": t["trial"], "passed": t["passed"], "duration": t["duration"], "error": t["error"]}
                for t in trials
            ],
        }

    n_cases = len(per_case)
    agg_pass_at_4 = sum(v["pass_at_4"] for v in per_case.values()) / n_cases
    agg_pass_pow_4 = sum(v["pass_pow_4"] for v in per_case.values()) / n_cases
    total_pass = sum(v["n_pass"] for v in per_case.values())
    total_trials = sum(v["n_trials"] for v in per_case.values())

    return {
        "n_cases": n_cases,
        "n_trials_per_case": 4,
        "total_trials": total_trials,
        "total_pass": total_pass,
        "pass_at_1": total_pass / total_trials,
        "pass_at_4": agg_pass_at_4,
        "pass_pow_4": agg_pass_pow_4,
        "per_case": per_case,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate all cases × 4 trials concurrently")
    parser.add_argument("--model", default=None, help="Model ID (default: $BELLA_MODEL_ID)")
    parser.add_argument("--base-url", default=None, help="API base URL (default: $BELLA_BASE_URL)")
    parser.add_argument("--api-key", default=None, help="API key (default: $BELLA_API_KEY)")
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=16, help="Concurrent workers")
    parser.add_argument("--cases-dir", default="cases")
    parser.add_argument("--output-dir", default=None, help="Output dir (default: results/eval_{timestamp})")
    args = parser.parse_args()

    model_kwargs = {
        "model_id": args.model or os.environ.get("BELLA_MODEL_ID", "gpt-5.2"),
        "base_url": args.base_url or os.environ.get("BELLA_BASE_URL"),
        "api_key": args.api_key or os.environ.get("BELLA_API_KEY"),
        "max_context_tokens": args.max_context_tokens or int(os.environ.get("BELLA_MAX_CONTEXT_TOKENS", "128000")),
    }

    with open("category_prompts.json") as f:
        category_prompts = json.load(f)

    environments_dir = Path("environments")
    cases_dir = Path(args.cases_dir)
    case_files = sorted(cases_dir.glob("tau3_airline_*.json"))
    cases = []
    for cf in case_files:
        with open(cf) as f:
            cases.append(json.load(f))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"results/eval_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model:   {args.model}")
    print(f"URL:     {args.base_url}")
    print(f"Cases:   {len(cases)}")
    print(f"Trials:  {args.n_trials}")
    print(f"Workers: {args.max_workers}")
    print(f"Output:  {output_dir}")
    print()

    eval_start = time.time()
    all_trial_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {}
        for case in cases:
            for trial in range(args.n_trials):
                fut = pool.submit(
                    run_single_trial,
                    case,
                    trial,
                    model_kwargs,
                    category_prompts,
                    environments_dir,
                    output_dir,
                )
                futures[fut] = (case["case_id"], trial)

        for fut in as_completed(futures):
            case_id, trial = futures[fut]
            try:
                result = fut.result()
                all_trial_results.append(result)
            except Exception as e:
                print(f"  [FATAL] {case_id}/trial{trial}: {e}", file=sys.stderr)
                all_trial_results.append({
                    "case_id": case_id,
                    "trial": trial,
                    "passed": False,
                    "error": str(e),
                    "duration": 0,
                    "tool_calls": 0,
                    "ended_normally": False,
                    "token_usage": None,
                    "verify": [],
                })

    eval_duration = time.time() - eval_start

    case_trials: dict[str, list[dict]] = {}
    for r in all_trial_results:
        case_trials.setdefault(r["case_id"], []).append(r)
    for trials in case_trials.values():
        trials.sort(key=lambda t: t["trial"])

    metrics = compute_metrics(case_trials)
    metrics["model"] = args.model
    metrics["base_url"] = args.base_url
    metrics["eval_duration"] = eval_duration
    metrics["timestamp"] = timestamp

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE  ({eval_duration:.1f}s)")
    print(f"{'='*60}")
    print(f"  Model:      {args.model}")
    print(f"  Cases:      {metrics['n_cases']}")
    print(f"  Trials:     {metrics['total_trials']}")
    print(f"  pass@1:     {metrics['pass_at_1']:.3f}  ({metrics['total_pass']}/{metrics['total_trials']})")
    print(f"  pass@4:     {metrics['pass_at_4']:.3f}  ({sum(1 for v in metrics['per_case'].values() if v['pass_at_4'])}/{metrics['n_cases']})")
    print(f"  pass^4:     {metrics['pass_pow_4']:.3f}  ({sum(1 for v in metrics['per_case'].values() if v['pass_pow_4'])}/{metrics['n_cases']})")
    print()

    print("Per-case breakdown:")
    for case_id, info in sorted(metrics["per_case"].items()):
        trials_str = "".join("P" if t["passed"] else "F" for t in info["trials"])
        print(f"  {case_id}  [{trials_str}]  {info['n_pass']}/{info['n_trials']}")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
