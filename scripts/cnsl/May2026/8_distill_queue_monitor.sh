#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

STATE_DIR="reports/cl_distil_eval/queue_2026-05-11"
mkdir -p "$STATE_DIR" logs/train/manual

GPU_LIST=(0 2 3)
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"

LABELS=(
  "conf2_light_replay40"
  "conf2_light_lr5e6"
  "conf2_light_drop005"
  "conf2_light_rank8"
  "conf2_light_rank8_replay40"
  "conf2_light_rank16"
  "conf2_light_rank12"
  "conf2_light_rank16_alpha16"
  "conf2_light_rank16_drop005"
  "conf2_light_rank16_replay40"
)

EXPERIMENTS=(
  "cnsl/May2026/10th_xlsr_conformertcm_mdt_cl-distill-conf-2-light-replay40"
  "cnsl/May2026/10th_xlsr_conformertcm_mdt_cl-distill-conf-2-light-lr5e-6"
  "cnsl/May2026/10th_xlsr_conformertcm_mdt_cl-distill-conf-2-light-drop005"
  "cnsl/May2026/10th_xlsr_conformertcm_mdt_cl-distill-conf-2-light-rank8"
  "cnsl/May2026/10th_xlsr_conformertcm_mdt_cl-distill-conf-2-light-rank8-replay40"
  "cnsl/May2026/10th_xlsr_conformertcm_mdt_cl-distill-conf-2-light-rank16"
  "cnsl/May2026/10th_xlsr_conformertcm_mdt_cl-distill-conf-2-light-rank12"
  "cnsl/May2026/10th_xlsr_conformertcm_mdt_cl-distill-conf-2-light-rank16-alpha16"
  "cnsl/May2026/10th_xlsr_conformertcm_mdt_cl-distill-conf-2-light-rank16-drop005"
  "cnsl/May2026/10th_xlsr_conformertcm_mdt_cl-distill-conf-2-light-rank16-replay40"
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$STATE_DIR/monitor.log"
}

init_state() {
  local label="$1"
  local status_file="$STATE_DIR/${label}.status"
  if [[ ! -f "$status_file" ]]; then
    printf 'queued\n' > "$status_file"
  fi
}

gpu_free() {
  local gpu used
  for gpu in "${GPU_LIST[@]}"; do
    if [[ -e "$STATE_DIR/gpu_${gpu}.reserved" ]]; then
      continue
    fi
    used="$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 1000 )); then
      printf '%s\n' "$gpu"
      return 0
    fi
  done
  return 1
}

has_queued() {
  local label status
  for label in "${LABELS[@]}"; do
    status="$(head -n 1 "$STATE_DIR/${label}.status" 2>/dev/null || true)"
    if [[ "$status" == "queued" ]]; then
      return 0
    fi
  done
  return 1
}

for label in "${LABELS[@]}"; do
  init_state "$label"
done

log "monitor start"

while has_queued; do
  for i in "${!LABELS[@]}"; do
    label="${LABELS[$i]}"
    exp="${EXPERIMENTS[$i]}"
    status_file="$STATE_DIR/${label}.status"
    status="$(head -n 1 "$status_file" 2>/dev/null || true)"
    [[ "$status" == "queued" ]] || continue

    if ! gpu="$(gpu_free)"; then
      log "no free GPU; wait"
      break
    fi

    session="cld_${label}_$(date '+%Y%m%d_%H%M%S')"
    train_log="logs/train/manual/${session}.log"
    printf '%s\n' "$label" > "$STATE_DIR/gpu_${gpu}.reserved"
    printf 'running gpu=%s session=%s exp=%s\n' "$gpu" "$session" "$exp" > "$status_file"
    log "launch $label on gpu $gpu session $session"

    tmux new-session -d -s "$session" \
      "cd '$REPO_ROOT' && bash scripts/cnsl/May2026/9_distill_queue_worker.sh '$label' '$gpu' '$exp' '$status_file' '$train_log'"

    # Let CUDA memory reservation become visible before considering another job.
    break
  done
  sleep "$SLEEP_SECONDS"
done

log "monitor done: no queued jobs"
