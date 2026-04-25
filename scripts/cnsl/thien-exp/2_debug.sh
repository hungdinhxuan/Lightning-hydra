#!/bin/bash
#
# Short profiling run for bottleneck investigation.
#
# Usage:
#   export XLSR_PRETRAINED_MODEL_PATH=/path/to/xlsr_ssl_checkpoint.pt
#   bash scripts/cnsl/thien-exp/2_debug.sh
#   bash scripts/cnsl/thien-exp/2_debug.sh -d 0

set -euo pipefail

while getopts "d:" opt; do
  case $opt in
    d) CUDA_DEVICE="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
  esac
done

LIMIT_TRAIN_BATCHES=${LIMIT_TRAIN_BATCHES:-100}
LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-20}
VAL_CHECK_INTERVAL=${VAL_CHECK_INTERVAL:-1.0}
OMP_THREADS=${OMP_THREADS:-16}
NUM_WORKERS=${NUM_WORKERS:-16}
PIN_MEMORY=${PIN_MEMORY:-true}
BATCH_SIZE=${BATCH_SIZE:-16}
EXTRA_ARGS_STR=${EXTRA_ARGS_STR:-}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${CUDA_DEVICE:-}" ]]; then
  CUDA_DEVICE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2,2n | head -n 1 | cut -d',' -f1 | tr -d ' ')
fi

EXTRA_ARGS=()
if [[ -n "$EXTRA_ARGS_STR" ]]; then
  read -r -a EXTRA_ARGS <<< "$EXTRA_ARGS_STR"
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=$OMP_THREADS uv run python src/train.py \
  debug=profiler \
  hydra.job_logging.root.level=INFO \
  experiment=thien_exp/xlsr_conformertcm_mdt_lora \
  ++data.data_dir="/dev/shm/April16_2026_Thien_recording" \
  ++data.args.protocol_path="/dev/shm/April16_2026_Thien_recording/protocol_train_dev_eval.txt" \
  ++train=True ++test=False \
  trainer.accelerator=cuda trainer.devices=1 \
  ++data.batch_size=$BATCH_SIZE \
  ++data.num_workers=$NUM_WORKERS ++data.pin_memory=$PIN_MEMORY \
  trainer.detect_anomaly=False \
  logger=csv +trainer.precision=bf16-mixed \
  ++trainer.limit_train_batches=$LIMIT_TRAIN_BATCHES \
  ++trainer.limit_val_batches=$LIMIT_VAL_BATCHES \
  ++trainer.val_check_interval=$VAL_CHECK_INTERVAL \
  ++model.is_base_model_path_ln=False \
  ++model.base_model_path="pretrained/MDT_241214_lora_250501.pt" \
  "${EXTRA_ARGS[@]}"
