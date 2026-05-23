#!/bin/bash
#
# XLSR-SLS full training, based on scripts/cnsl/May150526/2_full.sh.
#
# Usage:
#   export XLSR_PRETRAINED_MODEL_PATH=/path/to/xlsr_ssl_checkpoint.pt
#   bash scripts/cnsl/May150526/2_full_xlsr_sls.sh
#   bash scripts/cnsl/May150526/2_full_xlsr_sls.sh -d 0

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

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=16 python src/train.py \
  experiment=cnsl/May2026/17th_xlsr_sls_mdt \
  trainer=default \
  ++data.data_dir="/data/Datasets/dsd_corpus_pool_13May2026" \
  ++data.args.protocol_path="/data/Datasets/dsd_corpus_pool_13May2026/15_May_full.txt" \
  logger=wandb +trainer.precision=bf16-mixed \
  ++trainer.val_check_interval=0.25 ++model_averaging=True
