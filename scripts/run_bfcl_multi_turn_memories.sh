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

echo "All BFCL multi-turn inference jobs finished."
echo "Logs: $BELLA_BATCH_LOG_ROOT"
echo "Outputs: $BELLA_BATCH_OUTPUT_ROOT"
