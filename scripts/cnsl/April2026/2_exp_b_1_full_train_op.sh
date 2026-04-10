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

PROTOCOL_PATH=${LARGE_CORPUS_FOR_CNSL_PROTOCOLS:-data/protocols/cnsl/new_protocol_trim_vocoded_cleaned_v4_corrected.txt}
BATCH_SIZE=${BATCH_SIZE:-16}
VAL_CHECK_FRACTION=${VAL_CHECK_FRACTION:-0.5}

if [[ ! -f "$PROTOCOL_PATH" ]]; then
  echo "Protocol file not found: $PROTOCOL_PATH" >&2
  exit 1
fi

TRAIN_SAMPLES=$(python3 - "$PROTOCOL_PATH" <<'PY'
import shlex
import sys

protocol_path = sys.argv[1]
count = 0
with open(protocol_path, "r", encoding="utf-8") as f:
    for line in f:
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = shlex.split(raw)
        if len(parts) < 3:
            continue
        subset = parts[1].lower()
        if subset == "train":
            count += 1
print(count)
PY
)

TRAIN_EPOCH_BATCHES=$(( TRAIN_SAMPLES / BATCH_SIZE ))
if [[ "$TRAIN_EPOCH_BATCHES" -lt 1 ]]; then
  TRAIN_EPOCH_BATCHES=1
fi

VAL_CHECK_INTERVAL=$(python3 - "$TRAIN_EPOCH_BATCHES" "$VAL_CHECK_FRACTION" <<'PY'
import math
import sys

train_batches = int(sys.argv[1])
fraction = float(sys.argv[2])
k = max(1, int(math.floor(train_batches * fraction)))
print(k)
PY
)

echo "Protocol: $PROTOCOL_PATH"
echo "Train samples: $TRAIN_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Train epoch batches: $TRAIN_EPOCH_BATCHES"
echo "Validation check interval (batches): $VAL_CHECK_INTERVAL"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=5 python src/train.py \
  experiment=cnsl/April2026/exp_b_xlsr_conformertcm_mbct_mdt_optimized \
  logger=csv \
  ++data.args.wds_train_epoch_batches=$TRAIN_EPOCH_BATCHES \
  ++trainer.val_check_interval=$VAL_CHECK_INTERVAL \
  ++model_averaging=True
