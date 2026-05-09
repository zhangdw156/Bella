#!/usr/bin/env python3
"""Concurrent evaluation of cases. Computes pass@1, pass@k, pass^k."""

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from bella.model.anthropic import AnthropicModel
from bella.model.openai_chat import OpenAIChatModel
from bella.runner.bella import BellaRunner
from bella.types import Case

load_dotenv()

MODEL_CLASSES = {"anthropic": AnthropicModel, "openai_chat": OpenAIChatModel}


def _make_model(prefix: str = "BELLA") -> OpenAIChatModel | AnthropicModel:
    protocol = os.environ.get(f"{prefix}_MODEL_PROTOCOL", "openai_chat")
    cls = MODEL_CLASSES[protocol]
    return cls(
        model_id=os.environ.get(f"{prefix}_MODEL_ID", "gpt-5.2"),
        base_url=os.environ.get(f"{prefix}_BASE_URL"),
        api_key=os.environ.get(f"{prefix}_API_KEY"),
        max_context_tokens=int(os.environ.get(f"{prefix}_MAX_CONTEXT_TOKENS", "128000")),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate cases concurrently")
    parser.add_argument("--n", type=int, default=4, help="Number of trials per case")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--cases-dir", default="cases")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cases = Case.load_dir(Path(args.cases_dir))
    react_model = _make_model("BELLA")
    user_model = _make_model("BELLA_USER") if os.environ.get("BELLA_USER_MODEL_ID") else _make_model("BELLA")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"results/eval_{timestamp}")

    runner = BellaRunner(
        react_model=react_model,
        user_model=user_model,
        cases=cases,
        n=args.n,
        concurrency=args.concurrency,
        output_dir=output_dir,
    )
    result = runner.run()
    result.print_summary()


if __name__ == "__main__":
    main()
