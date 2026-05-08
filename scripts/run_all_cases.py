"""Run all cases and replay them."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from bella.model.anthropic import AnthropicModel
from bella.agent.react import ReactAgent
from bella.agent.user import UserAgent
from bella.runner.case import CaseRunner
from bella.runner.replay import ReplayRunner

load_dotenv()


def main():
    model_kwargs = {
        "model_id": os.environ.get("BELLA_MODEL_ID", "claude-opus-4.6"),
        "base_url": os.environ.get("BELLA_BASE_URL"),
        "api_key": os.environ.get("BELLA_API_KEY"),
        "max_context_tokens": int(os.environ.get("BELLA_MAX_CONTEXT_TOKENS", "128000")),
    }

    with open("category_prompts.json") as f:
        category_prompts = json.load(f)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    case_files = sorted(Path("cases").glob("tau3_airline_*.json"))
    replay_runner = ReplayRunner(environments_dir=Path("environments"))

    for case_file in case_files:
        with open(case_file) as f:
            case = json.load(f)

        print(f"\n{'='*60}")
        print(f"Case: {case['case_id']}")
        print(f"Demand: {case['demand'][:100]}...")
        print(f"{'='*60}")

        react_model = AnthropicModel(**model_kwargs)
        user_model = AnthropicModel(**model_kwargs)
        react_agent = ReactAgent(model=react_model, max_llm_calls_per_turn=12)
        user_agent = UserAgent(model=user_model, max_turns=30)

        runner = CaseRunner(
            react_agent=react_agent,
            user_agent=user_agent,
            environments_dir=Path("environments"),
            category_prompts=category_prompts,
            max_turns=30,
        )

        try:
            result = runner.run(case)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"  Run: {len(result.tool_calls)} tool calls, {result.timing.duration:.1f}s, ended_normally={result.ended_normally}")

        output = {
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
                    "tool_calls": [{"name": tc.name, "arguments": tc.arguments, "id": tc.id} for tc in m.tool_calls] if m.tool_calls else None,
                    "tool_call_id": m.tool_call_id,
                    "reasoning_content": m.reasoning_content,
                }
                for m in result.messages
            ],
            "token_usage": {"input_tokens": result.token_usage.input_tokens, "output_tokens": result.token_usage.output_tokens} if result.token_usage else None,
            "timing": {"duration": result.timing.duration},
        }

        case_result_file = results_dir / f"{result.case_id}.json"
        with open(case_result_file, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        replay = replay_runner.run(case, result.tool_calls)

        replay_output = {
            "case_id": case["case_id"],
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

        replay_result_file = results_dir / f"{result.case_id}_replay.json"
        with open(replay_result_file, "w") as f:
            json.dump(replay_output, f, indent=2, ensure_ascii=False)

        status = "PASS" if replay.passed else "FAIL"
        print(f"  Replay: {replay.matched}/{replay.total_calls} matched, {replay.token_substitutions} subs")
        print(f"  Verify: [{status}]")
        for vr in replay.verify_results:
            s = "PASS" if vr.passed else "FAIL"
            print(f"    [{s}] {vr.sql[:70]}")

    print(f"\n{'='*60}")
    print("Done. Results in results/")


if __name__ == "__main__":
    main()
