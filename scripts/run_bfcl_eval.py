#!/usr/bin/env python
from __future__ import annotations

import argparse

from bella.eval.bfcl_eval import run_bfcl_eval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BFCL evaluation via Bella using BFCL official evaluator."
    )
    parser.add_argument(
        "--category",
        type=str,
        default="simple_python",
        help="BFCL test category to evaluate (default: simple_python).",
    )
    parser.add_argument(
        "--no-partial-eval",
        action="store_true",
        help=(
            "Disable partial_eval flag in BFCL evaluator. "
            "Not recommended for MVP where only a subset is generated."
        ),
    )
    args = parser.parse_args()

    partial_eval = not args.no_partial_eval
    run_bfcl_eval(category=args.category, partial_eval=partial_eval)


if __name__ == "__main__":
    main()

