#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Tunables
: "${BELLA_ENTRY_WORKERS:=16}"
: "${BELLA_OUTER_JOBS:=2}"
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

run_category() {
  local memory_mode="$1"
  local category="$2"
  local registry_name="bella-multi-turn-${memory_mode}"
  local safe_registry_name="${registry_name//_/-}"
  local project_root="$BELLA_BATCH_OUTPUT_ROOT/${safe_registry_name}"
  local status_dir="$project_root/job_status"
  local status_file="$status_dir/${category}.json"
  local log_file="$BELLA_BATCH_LOG_ROOT/${safe_registry_name}-${category//_/-}.log"
  local infer_exit_code=0
  local eval_exit_code=0
  local final_status="ok"

  mkdir -p "$project_root" "$status_dir"

  {
    echo "[$(date '+%F %T')] START category=${category} memory=${memory_mode}"
    if env \
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
        --max-workers "$BELLA_ENTRY_WORKERS"; then
      if env \
        PYTHONUNBUFFERED=1 \
        BFCL_PROJECT_ROOT="$project_root" \
        BFCL_REGISTRY_NAME="$safe_registry_name" \
        .venv/bin/python scripts/run_bfcl_eval.py \
          --category "$category" \
          --no-partial-eval; then
        :
      else
        eval_exit_code=$?
        final_status="eval_failed"
        echo "[$(date '+%F %T')] ERROR evaluation failed with exit code ${eval_exit_code}"
      fi
    else
      infer_exit_code=$?
      final_status="infer_failed"
      echo "[$(date '+%F %T')] ERROR inference failed with exit code ${infer_exit_code}"
    fi

    cat >"$status_file" <<EOF
{"category":"$category","memory_mode":"$memory_mode","registry_name":"$safe_registry_name","status":"$final_status","infer_exit_code":$infer_exit_code,"eval_exit_code":$eval_exit_code}
EOF
    echo "[$(date '+%F %T')] END category=${category} memory=${memory_mode} status=${final_status}"
  } >"$log_file" 2>&1
}

run_memory() {
  local memory_mode="$1"
  local registry_name="bella-multi-turn-${memory_mode}"
  local safe_registry_name="${registry_name//_/-}"

  for category in "${MULTI_TURN_CATEGORIES[@]}"; do
    echo "Running memory=${memory_mode} category=${category} log=${BELLA_BATCH_LOG_ROOT}/${safe_registry_name}-${category//_/-}.log"
    run_category "$memory_mode" "$category"
  done
}

running_jobs=0

for memory_mode in "${MEMORY_MODES[@]}"; do
  run_memory "$memory_mode" &
  echo "Launched memory=${memory_mode} pid=$! output_root=${BELLA_BATCH_OUTPUT_ROOT}/bella-multi-turn-${memory_mode//_/-}"
  running_jobs=$((running_jobs + 1))

  if [ "$running_jobs" -ge "$BELLA_OUTER_JOBS" ]; then
    wait -n || true
    running_jobs=$((running_jobs - 1))
  fi
done

wait || true

BELLA_BATCH_OUTPUT_ROOT="$BELLA_BATCH_OUTPUT_ROOT" \
BELLA_SUMMARY_CSV="$summary_csv" \
BELLA_SUMMARY_JSON="$summary_json" \
.venv/bin/python - <<'PY'
from __future__ import annotations

import csv
import json
import os
from pathlib import Path


def load_score_header(score_path: Path) -> dict[str, object]:
    """BFCL score files are JSONL: first line is the summary header."""
    with score_path.open(encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        return {}
    return json.loads(first_line)

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
for memory_mode in memory_modes:
    registry_name = f"bella-multi-turn-{memory_mode}".replace("_", "-")
    for category in categories:
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
            "memory_mode": memory_mode,
            "category": category,
            "registry_name": registry_name,
            "score_file": str(score_path),
            "status": "missing",
            "infer_exit_code": "",
            "eval_exit_code": "",
            "accuracy": "",
            "accuracy_percent": "",
            "correct_count": "",
            "total_count": "",
        }
        status_path = output_root / registry_name / "job_status" / f"{category}.json"
        if status_path.exists():
            with status_path.open(encoding="utf-8") as f:
                status_payload = json.load(f)
            row["status"] = status_payload.get("status", "missing")
            row["infer_exit_code"] = status_payload.get("infer_exit_code", "")
            row["eval_exit_code"] = status_payload.get("eval_exit_code", "")
        if score_path.exists():
            header = load_score_header(score_path)
            accuracy = header.get("accuracy", "")
            row.update(
                {
                    "status": "ok",
                    "accuracy": accuracy,
                    "accuracy_percent": round(float(accuracy) * 100, 2)
                    if accuracy != ""
                    else "",
                    "correct_count": header.get("correct_count", ""),
                    "total_count": header.get("total_count", ""),
                }
            )
        rows.append(row)

with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "memory_mode",
            "category",
            "registry_name",
            "status",
            "infer_exit_code",
            "eval_exit_code",
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

echo "All BFCL multi-turn jobs finished."
echo "Logs: $BELLA_BATCH_LOG_ROOT"
echo "Outputs: $BELLA_BATCH_OUTPUT_ROOT"
echo "Summary CSV: $summary_csv"
echo "Summary JSON: $summary_json"
