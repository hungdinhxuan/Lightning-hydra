#!/bin/bash
#
# EXP D: foreground full training — MBCT + LoRA from pretrained A (S_241214_conf-1).
# Same overrides as 1_xlsr_conformertcm_mbct_lora.sh but uses exp_d experiment for tagging.
#
# Usage:
#   export XLSR_PRETRAINED_MODEL_PATH=/path/to/xlsr_ssl_checkpoint.pt
#   bash scripts/cnsl/April2026/exp_d_1_full_train.sh
#   bash scripts/cnsl/April2026/exp_d_1_full_train.sh -d 0

set -euo pipefail

while getopts "d:" opt; do
  case $opt in
    d) CUDA_DEVICE="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
  esac
done

CUDA_DEVICE=${CUDA_DEVICE:-"MIG-57de94a5-be15-5b5a-b67e-e118352d8a59"}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=32 python src/train.py \
  experiment=cnsl/April2026/exp_d_xlsr_conformertcm_mbct_mdt_lora \
  ++data.data_dir="data/DVC_DSD-Large-Corpus/raw/0_large-corpus_toys" \
  ++data.args.protocol_path="data/protocols/cnsl/new_protocol_trim_vocoded_cleaned_v4_corrected.txt" \
  logger=wandb \
  +trainer.val_check_interval=0.5 \
  ++model.is_base_model_path_ln=False \
  ++model.base_model_path="pretrained/S_241214_conf-1.pth" \
  ++model_averaging=True
