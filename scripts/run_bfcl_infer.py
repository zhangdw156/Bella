#!/usr/bin/env python
from __future__ import annotations

import argparse

from bella.infer.bfcl_runner import run_bfcl_infer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BFCL inference via Bella using BFCL official handlers."
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
        help="Maximum number of test entries to run (default: 3).",
    )
    args = parser.parse_args()

    run_bfcl_infer(category=args.category, limit=args.limit)


if __name__ == "__main__":
    main()

