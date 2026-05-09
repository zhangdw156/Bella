#!/usr/bin/env python3
"""Evaluate 5 retail cases with n=4, concurrency=5, gpt-5.2."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

from bella.model.openai_chat import OpenAIChatModel
from bella.runner.bella import BellaRunner
from bella.types import Case

load_dotenv()


CASE_IDS = {
    "tau3_retail_006",
    "tau3_retail_026",
    "tau3_retail_055",
    "tau3_retail_074",
    "tau3_retail_091",
    "tau3_retail_092",
    "tau3_retail_097",
    "tau3_retail_099",
    "tau3_retail_105",
}


def _make_model(prefix: str = "BELLA") -> OpenAIChatModel:
    return OpenAIChatModel(
        model_id=os.environ.get(f"{prefix}_MODEL_ID", "gpt-5.2"),
        base_url=os.environ.get(f"{prefix}_BASE_URL"),
        api_key=os.environ.get(f"{prefix}_API_KEY"),
        max_context_tokens=int(os.environ.get(f"{prefix}_MAX_CONTEXT_TOKENS", "128000")),
    )


def main():
    all_cases = Case.load_dir(Path("cases"))
    cases = [c for c in all_cases if c.case_id in CASE_IDS]
    print(f"Running {len(cases)} retail cases (n=4, concurrency=10, model=gpt-5.2)")
    for c in cases:
        print(f"  {c.case_id}")

    model = _make_model("BELLA")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/retail_eval_{timestamp}")

    runner = BellaRunner(
        react_model=model,
        user_model=model,
        cases=cases,
        n=4,
        concurrency=10,
        output_dir=output_dir,
    )
    result = runner.run()
    result.print_summary()


if __name__ == "__main__":
    main()
