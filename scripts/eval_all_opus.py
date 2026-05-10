#!/usr/bin/env python3
"""Evaluate all cases. react=claude-opus-4.5 (native Anthropic), user=gpt-5.2. n=4, concurrency=80."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

from bella.model.anthropic import AnthropicModel
from bella.model.openai_chat import OpenAIChatModel
from bella.runner.bella import BellaRunner
from bella.types import Case

load_dotenv()


def main():
    cases = Case.load_dir(Path("cases"))
    print(f"Loaded {len(cases)} cases")

    react_model = AnthropicModel(
        model_id="claude-opus-4.5",
        base_url=os.environ.get("ANTHROPIC_BASE_URL", "http://localhost:5152"),
        api_key=os.environ.get("ANTHROPIC_AUTH_TOKEN", os.environ.get("BELLA_API_KEY")),
        max_context_tokens=128000,
    )
    user_model = OpenAIChatModel(
        model_id="gpt-5.2",
        base_url=os.environ.get("BELLA_BASE_URL"),
        api_key=os.environ.get("BELLA_API_KEY"),
        max_context_tokens=128000,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/eval_all_opus_{timestamp}")

    runner = BellaRunner(
        react_model=react_model,
        user_model=user_model,
        cases=cases,
        n=4,
        concurrency=80,
        output_dir=output_dir,
    )
    result = runner.run()
    result.print_summary()


if __name__ == "__main__":
    main()
