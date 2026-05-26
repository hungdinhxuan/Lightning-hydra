#!/bin/bash
#
# Replay + MDT + LoRA + logit KD distillation.
#
# Usage:
#   bash scripts/cnsl/May2026/7_lora_replay_distill.sh -d 0
#   bash scripts/cnsl/May2026/7_lora_replay_distill.sh -d 0 -s
#   bash scripts/cnsl/May2026/7_lora_replay_distill.sh -d 0 -e cnsl/May2026/10th_xlsr_conformertcm_mdt_cl-distill-conf-2-light

set -euo pipefail

SMOKE=0
EXPERIMENT="cnsl/May2026/10th_xlsr_conformertcm_mdt_cl-distill-conf-4"
while getopts "d:e:s" opt; do
  case $opt in
    d) CUDA_DEVICE="$OPTARG";;
    e) EXPERIMENT="$OPTARG";;
    s) SMOKE=1;;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
  esac
done

CUDA_DEVICE=${CUDA_DEVICE:-"MIG-57de94a5-be15-5b5a-b67e-e118352d8a59"}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"/NAS1_pretrained_lab/29April26_xlsr_conformertcm_mdt_lora_merged.pth"}
TEACHER_CKPT_PATH=${TEACHER_CKPT_PATH:-"$BASE_MODEL_PATH"}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

EXTRA_ARGS=()
if [[ "$SMOKE" == "1" ]]; then
  EXTRA_ARGS+=(+trainer.fast_dev_run=2 ++data.num_workers=0 ++data.batch_size=4)
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=16 python src/train.py \
  experiment="$EXPERIMENT" \
  trainer=default \
  ++data.data_dir="/data/Datasets/dsd_corpus_pool_10May2026" \
  data.novel_protocol_path="/data/Datasets/dsd_corpus_pool_10May2026/novel_10_May_2026.txt" \
  data.replay_protocol_path="/data/Datasets/dsd_corpus_pool_10May2026/replay_april_dataset.txt" \
  data.return_source=true \
  test=false \
  logger=csv +trainer.precision=bf16-mixed \
  ++model.is_base_model_path_ln=False \
  ++model.base_model_path="$BASE_MODEL_PATH" \
  ++model.distill.teacher_ckpt_path="$TEACHER_CKPT_PATH" \
  "${EXTRA_ARGS[@]}"
