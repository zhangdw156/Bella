#!/usr/bin/env python3
"""Evaluate all cases. n=4, concurrency=80, model=gpt-5.2."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

from bella.model.openai_chat import OpenAIChatModel
from bella.runner.bella import BellaRunner
from bella.types import Case

load_dotenv()


def _make_model() -> OpenAIChatModel:
    return OpenAIChatModel(
        model_id="gpt-5.2",
        base_url=os.environ.get("BELLA_BASE_URL"),
        api_key=os.environ.get("BELLA_API_KEY"),
        max_context_tokens=128000,
    )


def main():
    cases = Case.load_dir(Path("cases"))
    print(f"Loaded {len(cases)} cases")

    react_model = _make_model()
    user_model = _make_model()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/eval_all_{timestamp}")

    runner = BellaRunner(
        react_model=react_model,
        user_model=user_model,
        cases=cases,
        # NOTE: n=4 is a good balance between discrimination and cost.
        # Experiments (2026-05-10) showed pass@1 has the best per-case discrimination (117/148),
        # pass^4 retains strong discrimination (108/148), while n=8 doubles cost with diminishing returns.
        n=4,
        concurrency=80,
        output_dir=output_dir,
    )
    result = runner.run()
    result.print_summary()


if __name__ == "__main__":
    main()
