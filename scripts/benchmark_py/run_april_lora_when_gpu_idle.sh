#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPU_ID="${1:-${GPU_ID:-3}}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
IDLE_GPU_UTIL_MAX="${IDLE_GPU_UTIL_MAX:-5}"
IDLE_MEM_USED_MAX_MB="${IDLE_MEM_USED_MAX_MB:-1024}"
IDLE_REQUIRED_CHECKS="${IDLE_REQUIRED_CHECKS:-3}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 127
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found" >&2
  exit 127
fi

if [[ -f ".venv/bin/activate" ]]; then
  # Keep shell environment aligned with this checkout while uv run handles execution.
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
else
  echo ".venv/bin/activate not found; continuing with uv run environment" >&2
fi

read_gpu_state() {
  nvidia-smi \
    --id="$GPU_ID" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits |
    tr -d ' '
}

idle_count=0
echo "Waiting for GPU ${GPU_ID} idle: util <= ${IDLE_GPU_UTIL_MAX}%, mem <= ${IDLE_MEM_USED_MAX_MB} MiB, checks = ${IDLE_REQUIRED_CHECKS}"

while true; do
  state="$(read_gpu_state)"
  util="${state%%,*}"
  mem="${state##*,}"
  now="$(date '+%F %T')"

  if [[ "$util" -le "$IDLE_GPU_UTIL_MAX" && "$mem" -le "$IDLE_MEM_USED_MAX_MB" ]]; then
    idle_count=$((idle_count + 1))
    echo "[$now] gpu=${GPU_ID} util=${util}% mem=${mem}MiB idle_check=${idle_count}/${IDLE_REQUIRED_CHECKS}"
  else
    idle_count=0
    echo "[$now] gpu=${GPU_ID} util=${util}% mem=${mem}MiB busy"
  fi

  if [[ "$idle_count" -ge "$IDLE_REQUIRED_CHECKS" ]]; then
    break
  fi

  sleep "$CHECK_INTERVAL_SECONDS"
done

echo "GPU ${GPU_ID} idle. Starting benchmark."

export DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-128}"

uv run ./scripts/benchmark_py/benchmark.py \
  -g "$GPU_ID" \
  -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer \
  -b data/April_2026_benchmark \
  -m /NAS1_pretrained_lab/06feb26_xlsr_conformertcm_mdt_vad.pt \
  -a /NAS1_pretrained_lab/lora/29April26_xlsr_conformertcm_mdt_lora_replay_from_06feb26_xlsr_conformertcm_mdt_vad_bf16-mixed \
  -r logs/results/April_2026_benchmark \
  -n "29April26_xlsr_conformertcm_mdt_lora_replay_from_06feb26_xlsr_conformertcm_mdt_vad_bf16-mixed" \
  -l false \
  +trainer.precision=bf16-mixed
