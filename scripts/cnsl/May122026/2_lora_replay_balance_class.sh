#!/bin/bash
#
# CNSL May 12 replay LoRA with class-balanced replay mini-batches.
#
# Usage:
#   bash scripts/cnsl/May122026/2_lora_replay_balance_class.sh
#   bash scripts/cnsl/May122026/2_lora_replay_balance_class.sh -d 0

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
  experiment=cnsl/May2026/12th_xlsr_conformertcm_mdt_cl-balance-class-conf-1.yaml \
  trainer=default \
  ++data.data_dir="/data/Datasets/dsd_corpus_pool_12May2026" \
  data.novel_protocol_path="/data/Datasets/dsd_corpus_pool_12May2026/novel_12_May_2026.txt" \
  data.replay_protocol_path="/data/Datasets/dsd_corpus_pool_12May2026/replay_april_dataset.txt" \
  logger=csv +trainer.precision=bf16-mixed \
  ++model.is_base_model_path_ln=False \
  ++model.base_model_path="/NAS1_pretrained_lab/29April26_xlsr_conformertcm_mdt_lora_merged.pth"
