#!/usr/bin/env python
from __future__ import annotations

import argparse

from bella.infer.runner import run_infer
from bella.eval.runner import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a minimal benchmark inference + evaluation demo via Bella."
    )
    parser.add_argument(
        "--benchmark", type=str, default="bfcl",
        help="Benchmark name (default: bfcl).",
    )
    parser.add_argument(
        "--category", type=str, default="simple_python",
        help="Category/subset to run (default: simple_python).",
    )
    parser.add_argument(
        "--limit", type=int, default=3,
        help="Maximum number of test entries to run in inference (default: 3).",
    )
    args = parser.parse_args()

    print(
        f"[Bella] Starting demo: benchmark={args.benchmark!r} "
        f"category={args.category!r} limit={args.limit}"
    )
    run_infer(args.benchmark, args.category, limit=args.limit)
    run_eval(args.benchmark, args.category, partial_eval=True)


if __name__ == "__main__":
    main()
