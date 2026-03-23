#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Tunables
: "${BELLA_ENTRY_WORKERS:=8}"
: "${BELLA_OUTER_JOBS:=4}"
: "${BELLA_OPENAI_TIMEOUT_SECONDS:=240}"
: "${BELLA_MULTI_TURN_MAX_STEPS_PER_TURN:=8}"
: "${BELLA_BATCH_OUTPUT_ROOT:=$ROOT_DIR/outputs/bfcl_batch}"
: "${BELLA_BATCH_LOG_ROOT:=$ROOT_DIR/logs/bfcl_batch}"
: "${BELLA_SUMMARY_BASENAME:=multi_turn_memory_summary}"

MEMORY_MODES=(
  "none"
  "action_history"
  "tool_result_memory"
  "tool_result_memory_v2"
)

MULTI_TURN_CATEGORIES=(
  "multi_turn_base"
  "multi_turn_miss_func"
  "multi_turn_miss_param"
  "multi_turn_long_context"
)

mkdir -p "$BELLA_BATCH_OUTPUT_ROOT" "$BELLA_BATCH_LOG_ROOT"

summary_csv="$BELLA_BATCH_OUTPUT_ROOT/${BELLA_SUMMARY_BASENAME}.csv"
summary_json="$BELLA_BATCH_OUTPUT_ROOT/${BELLA_SUMMARY_BASENAME}.json"

run_one() {
  local category="$1"
  local memory_mode="$2"
  local registry_name="bella-${category}-${memory_mode}"
  local safe_registry_name="${registry_name//_/-}"
  local project_root="$BELLA_BATCH_OUTPUT_ROOT/${safe_registry_name}"
  local log_file="$BELLA_BATCH_LOG_ROOT/${safe_registry_name}.log"

  mkdir -p "$project_root"

  {
    echo "[$(date '+%F %T')] START category=${category} memory=${memory_mode}"
    env \
      PYTHONUNBUFFERED=1 \
      BELLA_OPENAI_TIMEOUT_SECONDS="$BELLA_OPENAI_TIMEOUT_SECONDS" \
      BELLA_MULTI_TURN_MAX_STEPS_PER_TURN="$BELLA_MULTI_TURN_MAX_STEPS_PER_TURN" \
      BFCL_PROJECT_ROOT="$project_root" \
      BFCL_REGISTRY_NAME="$safe_registry_name" \
      BELLA_MULTI_TURN_MEMORY_MODE="$memory_mode" \
      BELLA_BFCL_MAX_WORKERS="$BELLA_ENTRY_WORKERS" \
      .venv/bin/python scripts/run_bfcl_infer.py \
        --category "$category" \
        --limit 0 \
        --max-workers "$BELLA_ENTRY_WORKERS"
    env \
      PYTHONUNBUFFERED=1 \
      BFCL_PROJECT_ROOT="$project_root" \
      BFCL_REGISTRY_NAME="$safe_registry_name" \
      .venv/bin/python scripts/run_bfcl_eval.py \
        --category "$category" \
        --no-partial-eval
    echo "[$(date '+%F %T')] END category=${category} memory=${memory_mode}"
  } >"$log_file" 2>&1
}

running_jobs=0

for category in "${MULTI_TURN_CATEGORIES[@]}"; do
  for memory_mode in "${MEMORY_MODES[@]}"; do
    run_one "$category" "$memory_mode" &
    running_jobs=$((running_jobs + 1))

    if [ "$running_jobs" -ge "$BELLA_OUTER_JOBS" ]; then
      wait -n
      running_jobs=$((running_jobs - 1))
    fi
  done
done

wait

BELLA_BATCH_OUTPUT_ROOT="$BELLA_BATCH_OUTPUT_ROOT" \
BELLA_SUMMARY_CSV="$summary_csv" \
BELLA_SUMMARY_JSON="$summary_json" \
.venv/bin/python - <<'PY'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

output_root = Path(os.environ["BELLA_BATCH_OUTPUT_ROOT"])
summary_csv = Path(os.environ["BELLA_SUMMARY_CSV"])
summary_json = Path(os.environ["BELLA_SUMMARY_JSON"])

memory_modes = [
    "none",
    "action_history",
    "tool_result_memory",
    "tool_result_memory_v2",
]
categories = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]
score_filenames = {
    "multi_turn_base": "BFCL_v4_multi_turn_base_score.json",
    "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func_score.json",
    "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param_score.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context_score.json",
}

rows: list[dict[str, object]] = []
for category in categories:
    for memory_mode in memory_modes:
        registry_name = f"bella-{category}-{memory_mode}".replace("_", "-")
        score_path = (
            output_root
            / registry_name
            / "score"
            / "__bella_isolated_eval__"
            / registry_name
            / registry_name
            / "multi_turn"
            / score_filenames[category]
        )
        row: dict[str, object] = {
            "category": category,
            "memory_mode": memory_mode,
            "registry_name": registry_name,
            "score_file": str(score_path),
            "status": "missing",
            "accuracy": "",
            "correct_count": "",
            "total_count": "",
        }
        if score_path.exists():
            with score_path.open(encoding="utf-8") as f:
                payload = json.load(f)
            header = payload[0] if isinstance(payload, list) and payload else {}
            accuracy = header.get("accuracy", "")
            row.update(
                {
                    "status": "ok",
                    "accuracy": accuracy,
                    "correct_count": header.get("correct_count", ""),
                    "total_count": header.get("total_count", ""),
                    "accuracy_percent": round(float(accuracy) * 100, 2)
                    if accuracy != ""
                    else "",
                }
            )
        rows.append(row)

with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "category",
            "memory_mode",
            "registry_name",
            "status",
            "accuracy",
            "accuracy_percent",
            "correct_count",
            "total_count",
            "score_file",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

with summary_json.open("w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(f"Summary CSV: {summary_csv}")
print(f"Summary JSON: {summary_json}")
PY

echo "All BFCL multi-turn inference jobs finished."
echo "Logs: $BELLA_BATCH_LOG_ROOT"
echo "Outputs: $BELLA_BATCH_OUTPUT_ROOT"
echo "Summary CSV: $summary_csv"
echo "Summary JSON: $summary_json"
