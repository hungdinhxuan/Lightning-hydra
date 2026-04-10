#!/bin/bash
#
# EXP D: full-strength MBCT + LoRA (pretrained A), detached log + PID (trial-experiment-monitoring).
# Same contract as 2_mbct_lora_full_strength_nohup.sh but experiment=exp_d_* for WandB tags.
#
# Usage:
#   export XLSR_PRETRAINED_MODEL_PATH=pretrained/xlsr2_300m.pt
#   bash scripts/cnsl/April2026/exp_d_2_full_strength_nohup.sh
#   bash scripts/cnsl/April2026/exp_d_2_full_strength_nohup.sh -d 0

set -euo pipefail

while getopts "d:" opt; do
  case $opt in
    d) CUDA_DEVICE="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
  esac
done

CUDA_DEVICE=${CUDA_DEVICE:-0}

if [[ -z "${XLSR_PRETRAINED_MODEL_PATH:-}" ]]; then
  echo "ERROR: export XLSR_PRETRAINED_MODEL_PATH" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs/train/exp_D_full_strength_mbct_lora"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_${STAMP}.log"
PID_FILE="$LOG_DIR/train_${STAMP}.pid"

nohup env \
  "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE" \
  OMP_NUM_THREADS=5 \
  "XLSR_PRETRAINED_MODEL_PATH=$XLSR_PRETRAINED_MODEL_PATH" \
  uv run python src/train.py \
  experiment=cnsl/April2026/exp_d_xlsr_conformertcm_mbct_lora \
  ++data.data_dir=data/DVC_DSD-Large-Corpus/raw/0_large-corpus_toys \
  ++data.args.protocol_path=data/protocols/cnsl/new_protocol_trim_vocoded_cleaned_v4_corrected.txt \
  logger=wandb \
  +trainer.val_check_interval=0.5 \
  ++model_averaging=True \
  ++model.is_base_model_path_ln=False \
  ++model.base_model_path=pretrained/S_241214_conf-1.pth \
  ++data.args.enable_cache=false \
  >"$LOG_FILE" 2>&1 &

echo $! >"$PID_FILE"

echo "Started PID $(cat "$PID_FILE")"
echo "Log: $LOG_FILE"
echo "PID file: $PID_FILE"
echo "Tail: tail -f $LOG_FILE"
