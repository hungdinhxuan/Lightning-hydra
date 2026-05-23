#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
MAX_ROUNDS="${MAX_ROUNDS:-0}"
AGENT_BIN="${AGENT_BIN:-/home/hungdx/.nvm/versions/node/v22.14.0/bin/codex}"
STATE_DIR="reports/cl_distil_eval/auto_agent_2026-05-11"
mkdir -p "$STATE_DIR" logs/agent

round=0
while true; do
  round=$((round + 1))
  ts="$(date '+%Y%m%d_%H%M%S')"
  log_file="logs/agent/cl_distill_auto_agent_${ts}.jsonl"
  msg_file="logs/agent/cl_distill_auto_agent_${ts}.final.txt"

  printf '[%s] auto-agent round %s start\n' "$(date '+%F %T')" "$round" | tee -a "$STATE_DIR/loop.log"

  "$AGENT_BIN" exec \
    --cd "$REPO_ROOT" \
    --sandbox danger-full-access \
    --dangerously-bypass-approvals-and-sandbox \
    --json \
    --output-last-message "$msg_file" \
    - < scripts/cnsl/May2026/auto_improve_prompt.md > "$log_file" 2>&1 || true

  printf '[%s] auto-agent round %s done log=%s final=%s\n' "$(date '+%F %T')" "$round" "$log_file" "$msg_file" | tee -a "$STATE_DIR/loop.log"

  if [[ "$MAX_ROUNDS" != "0" && "$round" -ge "$MAX_ROUNDS" ]]; then
    printf '[%s] max rounds reached\n' "$(date '+%F %T')" | tee -a "$STATE_DIR/loop.log"
    exit 0
  fi

  sleep "$INTERVAL_SECONDS"
done
