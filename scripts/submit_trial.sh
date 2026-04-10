#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/submit_trial.sh <trial_id> <experiment_cfg> [priority]

QUEUE_FILE="${QUEUE_FILE:-plans/01042026/queue.jsonl}"
TRIAL_ID="${1:-}"
EXPERIMENT_CFG="${2:-}"
PRIORITY="${3:-5}"

if [[ -z "$TRIAL_ID" || -z "$EXPERIMENT_CFG" ]]; then
  echo "Usage: scripts/submit_trial.sh <trial_id> <experiment_cfg> [priority]"
  exit 1
fi

COMMAND="python src/train.py experiment=${EXPERIMENT_CFG}"
printf '{"trial_id":"%s","owner":"%s","experiment":"%s","status":"pending","priority":%s,"gpu_id":null,"batch_size":96,"command":"%s","updated_at":"%s"}\n' \
  "$TRIAL_ID" "${USER:-unknown}" "$EXPERIMENT_CFG" "$PRIORITY" "$COMMAND" "$(date -u +%FT%TZ)" >> "$QUEUE_FILE"

echo "queued: ${TRIAL_ID}"

