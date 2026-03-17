#!/usr/bin/env python
from __future__ import annotations

import argparse

from bella.infer.bfcl_runner import run_bfcl_infer
from bella.eval.bfcl_eval import run_bfcl_eval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a minimal BFCL inference + evaluation demo via Bella."
    )
    parser.add_argument(
        "--category",
        type=str,
        default="simple_python",
        help="BFCL test category to run (default: simple_python).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of test entries to run in inference (default: 3).",
    )
    args = parser.parse_args()

    print(
        f"[Bella] Starting BFCL demo with category='{args.category}', "
        f"limit={args.limit}."
    )
    run_bfcl_infer(category=args.category, limit=args.limit)
    run_bfcl_eval(category=args.category, partial_eval=True)


if __name__ == "__main__":
    main()

