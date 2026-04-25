#!/usr/bin/env bash
set -euo pipefail

# Minimal trigger script for cron:
# 1) check for pending tasks
# 2) check for free GPU
# 3) start agent/workflow only when both are true

QUEUE_FILE="${QUEUE_FILE:-queue.jsonl}"
LOCK_FILE="${LOCK_FILE:-/tmp/ml-exp-orchestrator.lock}"

if [[ -f "$LOCK_FILE" ]]; then
  echo "lock exists, skipping"
  exit 0
fi

touch "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

has_pending=$(rg -n '"status"\s*:\s*"pending"' "$QUEUE_FILE" >/dev/null && echo "yes" || echo "no")
has_gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '$1 < 500 {found=1} END{print found ? "yes" : "no"}')

if [[ "$has_pending" == "yes" && "$has_gpu" == "yes" ]]; then
  echo "trigger: pending task + free gpu"
  # Replace this command with your project-specific agent trigger.
  # Example:
  # cursor-agent run --task "pick next queue task and execute"
  agent --continue -p "pick next queue task and execute"
  echo "agent triggered"
  exit 0
else
  echo "no trigger condition met"
fi
