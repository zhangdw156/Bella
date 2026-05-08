"""Run a single case for debugging."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from bella.model.anthropic import AnthropicModel
from bella.agent.react import ReactAgent
from bella.agent.user import UserAgent
from bella.runner.case import CaseRunner

load_dotenv()


def main():
    model_kwargs = {
        "model_id": os.environ.get("BELLA_MODEL_ID", "claude-opus-4.6"),
        "base_url": os.environ.get("BELLA_BASE_URL"),
        "api_key": os.environ.get("BELLA_API_KEY"),
        "max_context_tokens": int(os.environ.get("BELLA_MAX_CONTEXT_TOKENS", "128000")),
    }

    react_model = AnthropicModel(**model_kwargs)
    user_model = AnthropicModel(**model_kwargs)

    react_agent = ReactAgent(model=react_model, max_llm_calls_per_turn=12)
    user_agent = UserAgent(model=user_model, max_turns=30)

    with open("category_prompts.json") as f:
        category_prompts = json.load(f)

    runner = CaseRunner(
        react_agent=react_agent,
        user_agent=user_agent,
        environments_dir=Path("environments"),
        category_prompts=category_prompts,
        max_turns=30,
    )

    case_file = Path("cases/tau3_airline_019.json")
    with open(case_file) as f:
        case = json.load(f)

    print(f"Running case: {case['case_id']} ({case['interaction_mode']} mode)")
    print(f"Demand: {case.get('demand', 'N/A')}")
    print("-" * 60)

    result = runner.run(case)

    print(f"\nCase completed.")
    print(f"  react model: {result.react_model_config['model_id']} ({result.react_model_config['model_class']})")
    if result.user_model_config:
        print(f"  user model: {result.user_model_config['model_id']} ({result.user_model_config['model_class']})")
    print(f"  ended_normally: {result.ended_normally}")
    print(f"  tool_calls: {len(result.tool_calls)}")
    print(f"  token_usage: {result.token_usage.total_tokens if result.token_usage else 'N/A'}")
    print(f"  duration: {result.timing.duration:.1f}s")

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
        "timing": {"start_time": result.timing.start_time, "end_time": result.timing.end_time, "duration": result.timing.duration},
    }

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{result.case_id}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  result saved to: {output_file}")


if __name__ == "__main__":
    main()
