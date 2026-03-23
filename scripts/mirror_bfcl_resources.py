#!/usr/bin/env python
"""
Mirror BFCL v4 datasets and multi_turn tool schemas into Bella so inference
does not depend on BFCL runtime data loader. Run from Bella repo root.

Usage:
  python -m scripts.mirror_bfcl_resources --bfcl-root /path/to/berkeley-function-call-leaderboard
  # or set env BFCL_PROJECT_ROOT
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# BFCL v4 category -> BFCL data filename (without path)
_DATASET_FILES = {
    "simple_python": "BFCL_v4_simple_python.json",
    "multiple": "BFCL_v4_multiple.json",
    "multi_turn_base": "BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
}

# involved_classes -> func doc filename (from MULTI_TURN_FUNC_DOC_FILE_MAPPING)
_MULTI_TURN_FUNC_DOC_FILES = [
    "gorilla_file_system.json",
    "math_api.json",
    "message_api.json",
    "posting_api.json",
    "ticket_api.json",
    "trading_bot.json",
    "travel_booking.json",
    "vehicle_control.json",
    "web_search.json",
    "memory_kv.json",
    "memory_vector.json",
    "memory_rec_sum.json",
]


def _bella_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").exists():
        raise FileNotFoundError(f"Bella root not found (expected pyproject.toml in {root})")
    return root


def mirror_datasets(bfcl_data_dir: Path, bella_raw_dir: Path) -> None:
    """Copy BFCL v4 JSONL datasets to datasets/bfcl_v4/raw/<category>.jsonl."""
    bella_raw_dir.mkdir(parents=True, exist_ok=True)
    for category, filename in _DATASET_FILES.items():
        src = bfcl_data_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"BFCL data file not found: {src}")
        dst = bella_raw_dir / f"{category}.jsonl"
        shutil.copy2(src, dst)
        print(f"  {src.name} -> {dst.relative_to(_bella_root())}")


def mirror_tool_schemas(bfcl_func_doc_dir: Path, bella_tool_schemas_dir: Path) -> None:
    """Copy multi_turn func doc JSONL files to tool_schemas/bfcl_v4/multi_turn/."""
    bella_tool_schemas_dir.mkdir(parents=True, exist_ok=True)
    for name in _MULTI_TURN_FUNC_DOC_FILES:
        src = bfcl_func_doc_dir / name
        if not src.exists():
            raise FileNotFoundError(f"BFCL func doc not found: {src}")
        shutil.copy2(src, bella_tool_schemas_dir / name)
        print(f"  {name} -> tool_schemas/bfcl_v4/multi_turn/{name}")


def main() -> None:
    import os
    parser = argparse.ArgumentParser(
        description="Mirror BFCL v4 datasets and multi_turn tool schemas into Bella."
    )
    parser.add_argument(
        "--bfcl-root",
        type=Path,
        default=None,
        help="Path to berkeley-function-call-leaderboard repo (default: env BFCL_PROJECT_ROOT).",
    )
    args = parser.parse_args()
    bfcl_root = args.bfcl_root
    if bfcl_root is None and os.environ.get("BFCL_PROJECT_ROOT"):
        bfcl_root = Path(os.environ["BFCL_PROJECT_ROOT"])
    if bfcl_root is None:
        bfcl_root = Path(__file__).resolve().parents[2] / "gorilla" / "berkeley-function-call-leaderboard"
    if not bfcl_root.exists():
        raise FileNotFoundError(
            f"BFCL root not found: {bfcl_root}. Set --bfcl-root or BFCL_PROJECT_ROOT."
        )

    root = _bella_root()
    bfcl_data = bfcl_root / "bfcl_eval" / "data"
    bfcl_func_doc = bfcl_data / "multi_turn_func_doc"
    raw_dir = root / "datasets" / "bfcl_v4" / "raw"
    tool_dir = root / "tool_schemas" / "bfcl_v4" / "multi_turn"

    print("Mirroring datasets...")
    mirror_datasets(bfcl_data, raw_dir)
    print("Mirroring multi_turn tool schemas...")
    mirror_tool_schemas(bfcl_func_doc, tool_dir)
    print("Done.")


if __name__ == "__main__":
    main()
