#!/bin/bash
#
# EXP B: foreground full training — MBCT, full model from SSL (no LoRA).
# trial-experiment-monitoring: logger=wandb, +trainer.val_check_interval, ++model_averaging=True
#
# Usage:
#   export XLSR_PRETRAINED_MODEL_PATH=/path/to/xlsr_ssl_checkpoint.pt
#   bash scripts/cnsl/April2026/exp_b_1_full_train.sh
#   bash scripts/cnsl/April2026/exp_b_1_full_train.sh -d 0

set -euo pipefail

while getopts "d:" opt; do
  case $opt in
    d) CUDA_DEVICE="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
  esac
done

CUDA_DEVICE=${CUDA_DEVICE:-0}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=5 python src/train.py \
  experiment=cnsl/April2026/exp_b_xlsr_conformertcm_mbct_mdt \
  ++data.data_dir="data/DVC_DSD-Large-Corpus/raw/0_large-corpus_toys" \
  ++data.args.protocol_path="data/protocols/cnsl/new_protocol_trim_vocoded_cleaned_v4_corrected.txt" \
  logger=wandb \
  +trainer.val_check_interval=0.5 \
  ++model_averaging=True
