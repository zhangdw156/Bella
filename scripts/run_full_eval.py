#!/usr/bin/env python
"""
Full evaluation matrix: (memory_mode × benchmark × category).

For the model configured in .env, runs inference + evaluation for:
  - BFCL multi-turn (4 categories) × {none, mem0}
  - LoCoMo QA × {none, mem0}  (mem0 ingests conversation into memory, retrieves per question)

Intermediate results  → outputs/
Score summaries       → artifacts/

Usage:
    python scripts/run_full_eval.py
    python scripts/run_full_eval.py --max-workers 8
    python scripts/run_full_eval.py --bfcl-only
    python scripts/run_full_eval.py --locomo-only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))


# ── data preparation ─────────────────────────────────────────────────

LOCOMO_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
LOCOMO_PATH = ROOT_DIR / "datasets" / "locomo" / "locomo10.json"


def ensure_locomo_data() -> None:
    if LOCOMO_PATH.exists():
        return
    print(f"[Bella] Downloading locomo10.json → {LOCOMO_PATH}")
    LOCOMO_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(LOCOMO_URL, LOCOMO_PATH)
    size_mb = LOCOMO_PATH.stat().st_size / 1024 / 1024
    print(f"[Bella] Downloaded {size_mb:.1f} MB")


# ── evaluation matrix ────────────────────────────────────────────────

MEMORY_MODES = ["none", "mem0"]

BFCL_CATEGORIES = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]

LOCOMO_CATEGORIES = ["qa"]


def _run_one(
    benchmark: str,
    category: str,
    memory_mode: str,
    max_workers: int,
) -> dict[str, Any]:
    """Run inference + evaluation for one (benchmark, category, memory_mode) cell.

    Returns a result dict for the summary table.
    """
    from bella.config import load_settings

    row: dict[str, Any] = {
        "benchmark": benchmark,
        "category": category,
        "memory_mode": memory_mode,
        "status": "pending",
        "score": "",
        "detail": "",
    }

    os.environ["BELLA_MULTI_TURN_MEMORY_MODE"] = memory_mode
    os.environ["BELLA_MAX_WORKERS"] = str(max_workers)

    if benchmark == "bfcl":
        safe_name = f"bella-eval-{memory_mode}".replace("_", "-")
        project_root = str(ROOT_DIR / "outputs" / "eval_batch" / f"bfcl_{memory_mode}")
        os.environ["BFCL_PROJECT_ROOT"] = project_root
        os.environ["BFCL_REGISTRY_NAME"] = safe_name
        os.makedirs(project_root, exist_ok=True)

    settings = load_settings()

    try:
        from bella.infer.runner import run_infer
        print(f"\n{'='*60}")
        print(f"[Bella] INFER  benchmark={benchmark}  category={category}  memory={memory_mode}")
        print(f"{'='*60}")
        run_infer(benchmark, category, limit=0, max_workers=max_workers)
    except Exception as e:
        row["status"] = "infer_failed"
        row["detail"] = str(e)
        print(f"[Bella] INFER FAILED: {e}")
        return row

    try:
        from bella.eval.runner import run_eval
        print(f"\n[Bella] EVAL   benchmark={benchmark}  category={category}  memory={memory_mode}")
        run_eval(benchmark, category, partial_eval=False)
        row["status"] = "ok"
    except Exception as e:
        row["status"] = "eval_failed"
        row["detail"] = str(e)
        print(f"[Bella] EVAL FAILED: {e}")
        return row

    row["score"] = _extract_score(benchmark, category, memory_mode)
    return row


def _extract_score(benchmark: str, category: str, memory_mode: str) -> str:
    """Best-effort score extraction after evaluation."""
    if benchmark == "locomo":
        score_file = ROOT_DIR / "outputs" / "locomo" / memory_mode / "scores" / f"locomo_{category}_score.json"
        if score_file.exists():
            data = json.loads(score_file.read_text(encoding="utf-8"))
            return str(data.get("overall", ""))

    if benchmark == "bfcl":
        safe_name = f"bella-eval-{memory_mode}".replace("_", "-")
        project_root = ROOT_DIR / "outputs" / "eval_batch" / f"bfcl_{memory_mode}"
        score_dir = project_root / "score" / "__bella_isolated_eval__" / safe_name / safe_name
        if score_dir.exists():
            for f in score_dir.rglob("*.json"):
                if category in f.name and "score" in f.name:
                    first_line = f.read_text(encoding="utf-8").strip().split("\n")[0]
                    try:
                        header = json.loads(first_line)
                        acc = header.get("accuracy", "")
                        if acc != "":
                            return str(round(float(acc) * 100, 2)) + "%"
                    except (json.JSONDecodeError, ValueError):
                        pass
    return ""


# ── summary generation ───────────────────────────────────────────────

def _write_summary(rows: list[dict[str, Any]], artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifacts_dir / "eval_summary.csv"
    json_path = artifacts_dir / "eval_summary.json"

    fieldnames = ["benchmark", "category", "memory_mode", "status", "score", "detail"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("[Bella] EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Benchmark':<10} {'Category':<28} {'Memory':<8} {'Status':<14} {'Score':<10}")
    print("-" * 72)
    for r in rows:
        print(
            f"{r['benchmark']:<10} {r['category']:<28} {r['memory_mode']:<8} "
            f"{r['status']:<14} {r['score']:<10}"
        )
    print("-" * 72)
    print(f"Summary CSV:  {csv_path}")
    print(f"Summary JSON: {json_path}")


# ── main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Full evaluation matrix for Bella.")
    parser.add_argument("--max-workers", type=int, default=1, help="Concurrent inference workers.")
    parser.add_argument("--bfcl-only", action="store_true", help="Only run BFCL benchmarks.")
    parser.add_argument("--locomo-only", action="store_true", help="Only run LoCoMo benchmarks.")
    args = parser.parse_args()

    run_bfcl = not args.locomo_only
    run_locomo = not args.bfcl_only

    if run_locomo:
        ensure_locomo_data()

    rows: list[dict[str, Any]] = []

    if run_bfcl:
        for memory_mode in MEMORY_MODES:
            for category in BFCL_CATEGORIES:
                row = _run_one("bfcl", category, memory_mode, args.max_workers)
                rows.append(row)

    if run_locomo:
        for memory_mode in MEMORY_MODES:
            for category in LOCOMO_CATEGORIES:
                row = _run_one("locomo", category, memory_mode, args.max_workers)
                rows.append(row)

    artifacts_dir = ROOT_DIR / "artifacts"
    _write_summary(rows, artifacts_dir)


if __name__ == "__main__":
    main()
