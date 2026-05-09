#!/usr/bin/env python3
"""Run a single case for debugging."""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from bella.model.anthropic import AnthropicModel
from bella.model.openai_chat import OpenAIChatModel
from bella.runner.bella import BellaRunner
from bella.types import Case

load_dotenv()

MODEL_CLASSES = {"anthropic": AnthropicModel, "openai_chat": OpenAIChatModel}


def _make_model(prefix: str = "BELLA") -> tuple:
    protocol = os.environ.get(f"{prefix}_MODEL_PROTOCOL", "openai_chat")
    cls = MODEL_CLASSES[protocol]
    return cls(
        model_id=os.environ.get(f"{prefix}_MODEL_ID", "gpt-5.2"),
        base_url=os.environ.get(f"{prefix}_BASE_URL"),
        api_key=os.environ.get(f"{prefix}_API_KEY"),
        max_context_tokens=int(os.environ.get(f"{prefix}_MAX_CONTEXT_TOKENS", "128000")),
    )


def main():
    case_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("cases/tau3_airline_019.json")
    case = Case.from_json(case_path)

    react_model = _make_model("BELLA")
    user_model = _make_model("BELLA_USER") if os.environ.get("BELLA_USER_MODEL_ID") else _make_model("BELLA")

    runner = BellaRunner(
        react_model=react_model,
        user_model=user_model,
        cases=[case],
        n=1,
        concurrency=1,
        output_dir=Path("results"),
    )
    result = runner.run()
    result.print_summary()


if __name__ == "__main__":
    main()
