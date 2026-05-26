#!/usr/bin/env bash

set -uo pipefail

LABEL="$1"
GPU="$2"
EXPERIMENT="$3"
STATUS_FILE="$4"
TRAIN_LOG="$5"
STATE_DIR="$(dirname "$STATUS_FILE")"

XLSR_PRETRAINED_MODEL_PATH="${XLSR_PRETRAINED_MODEL_PATH:-/nvme2/hungdx/Lightning-hydra/pretrained/xlsr2_300m.pt}"
export XLSR_PRETRAINED_MODEL_PATH

set +e
{
  printf '[%s] start %s gpu=%s exp=%s\n' "$(date '+%F %T')" "$LABEL" "$GPU" "$EXPERIMENT"
  printf '[%s] smoke\n' "$(date '+%F %T')"
  bash scripts/cnsl/May2026/7_lora_replay_distill.sh -d "$GPU" -e "$EXPERIMENT" -s
  printf '[%s] full train\n' "$(date '+%F %T')"
  bash scripts/cnsl/May2026/7_lora_replay_distill.sh -d "$GPU" -e "$EXPERIMENT"
  printf '[%s] done\n' "$(date '+%F %T')"
} > "$TRAIN_LOG" 2>&1
rc=$?
set -e

if [[ "$rc" -eq 0 ]]; then
  printf 'done gpu=%s exp=%s log=%s\n' "$GPU" "$EXPERIMENT" "$TRAIN_LOG" > "$STATUS_FILE"
else
  printf 'failed rc=%s gpu=%s exp=%s log=%s\n' "$rc" "$GPU" "$EXPERIMENT" "$TRAIN_LOG" > "$STATUS_FILE"
fi

rm -f "$STATE_DIR/gpu_${GPU}.reserved"

exit "$rc"
