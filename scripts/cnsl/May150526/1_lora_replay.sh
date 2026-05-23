#!/bin/bash
#
# EXP D foreground full training: MDT×MBCT LoRA — normal + narrowband (8 views).
# trial-experiment-monitoring: logger=wandb, +trainer.val_check_interval, ++model_averaging=True
#
# Usage:
#   export XLSR_PRETRAINED_MODEL_PATH=/path/to/xlsr_ssl_checkpoint.pt
#   bash scripts/cnsl/April2026/exp_d_1_full_train_2band_normal_narrowband.sh
#   bash scripts/cnsl/April2026/exp_d_1_full_train_2band_normal_narrowband.sh -d 0

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

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=8 python src/train.py \
  experiment=cnsl/May2026/15th_xlsr_conformertcm_mdt_cl \
  trainer=default \
  ++data.data_dir="/data/Datasets/dsd_corpus_pool_13May2026" \
  data.novel_protocol_path="/data/Datasets/dsd_corpus_pool_13May2026/novel_15_May_full.txt" \
  data.replay_protocol_path="/data/Datasets/dsd_corpus_pool_13May2026/replay_feb_dataset.txt" \
  logger=wandb +trainer.precision=bf16-mixed \
  ++model.is_base_model_path_ln=False \
  ++trainer.val_check_interval=0.5 \
  ++model.base_model_path="/NAS1_pretrained_lab/06feb26_xlsr_conformertcm_mdt_vad.pt"
