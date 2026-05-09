#!/usr/bin/env python3
"""Run one batch: migrate cases, evaluate, report results."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: str, timeout: int = 600) -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    return result.stdout


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/batch_eval.py ID1 ID2 ...")
        sys.exit(1)

    task_ids = set(sys.argv[1:])
    ids_str = '","'.join(sorted(task_ids, key=int))

    # Step 1: Update migration script and generate cases
    migrate_script = Path("scripts/migrate_tau2_retail_cases.py")
    text = migrate_script.read_text()
    import re
    text = re.sub(
        r'SELECTED_IDS: set\[str\] \| None = .*',
        f'SELECTED_IDS: set[str] | None = {{"{ids_str}"}}  # None = all mutating tasks',
        text,
    )
    migrate_script.write_text(text)

    print(f"=== Migrating {len(task_ids)} cases ===")
    print(run("uv run python scripts/migrate_tau2_retail_cases.py"))

    # Step 2: Update eval script
    eval_script = Path("scripts/eval_retail_5.py")
    text = eval_script.read_text()
    case_ids = [f"tau3_retail_{int(tid):03d}" for tid in sorted(task_ids, key=int)]
    case_ids_block = "CASE_IDS = {\n" + "".join(f'    "{cid}",\n' for cid in case_ids) + "}"
    text = re.sub(r'CASE_IDS = \{[^}]+\}', case_ids_block, text)
    eval_script.write_text(text)

    # Step 3: Run eval
    print(f"=== Evaluating {len(task_ids)} cases (n=4, concurrency=10) ===")
    output = run("uv run python scripts/eval_retail_5.py", timeout=600)
    print(output)

    # Step 4: Find latest results dir
    results_dirs = sorted(Path("results").glob("retail_eval_*"))
    if not results_dirs:
        print("No results found!")
        sys.exit(1)
    results_dir = results_dirs[-1]

    # Step 5: Analyze
    perfect = []
    zero = []
    for cid in case_ids:
        passes = 0
        for trial in range(4):
            rfile = results_dir / "runs" / f"{cid}_trial{trial}.json"
            if rfile.exists():
                with open(rfile) as f:
                    data = json.load(f)
                if data.get("passed"):
                    passes += 1
        if passes == 4:
            perfect.append(cid)
        elif passes == 0:
            zero.append(cid)

    # Step 6: Delete too-easy cases
    if perfect:
        print(f"\n=== Deleting {len(perfect)} too-easy cases (4/4) ===")
        for cid in perfect:
            p = Path(f"cases/{cid}.json")
            if p.exists():
                p.unlink()
                print(f"  Deleted {cid}")

    if zero:
        print(f"\n=== 0/4 cases (need investigation): {zero} ===")

    print("\nDone.")


if __name__ == "__main__":
    main()
