#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./submit_trial.sh <trial_id> <experiment_name> <command> [priority]

QUEUE_FILE="${QUEUE_FILE:-queue.jsonl}"
TRIAL_ID="${1:-}"
EXPERIMENT="${2:-}"
COMMAND="${3:-}"
PRIORITY="${4:-5}"

if [[ -z "$TRIAL_ID" || -z "$EXPERIMENT" || -z "$COMMAND" ]]; then
  echo "Usage: ./submit_trial.sh <trial_id> <experiment_name> <command> [priority]"
  exit 1
fi

printf '{"trial_id":"%s","owner":"%s","experiment":"%s","status":"pending","priority":%s,"gpu_id":null,"batch_size":96,"command":"%s","updated_at":"%s"}\n' \
  "$TRIAL_ID" "${USER:-unknown}" "$EXPERIMENT" "$PRIORITY" "$COMMAND" "$(date -u +%FT%TZ)" >> "$QUEUE_FILE"

echo "Enqueued ${TRIAL_ID} in ${QUEUE_FILE}"
