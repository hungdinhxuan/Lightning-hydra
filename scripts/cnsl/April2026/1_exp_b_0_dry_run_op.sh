#!/bin/bash
# EXP B: MBCT full-model smoke test (fast_dev_run), no LoRA — plans/20260402_mbct_experiment/EXP.md
#
# Uses default_loss_earlystop (no LR monitor) so +trainer.fast_dev_run works with loggers disabled.
# ++test=False: skip test (BaseLitModule needs score_save_path for test).
#
# Usage:
#   export XLSR_PRETRAINED_MODEL_PATH=pretrained/xlsr2_300m.pt
#   bash scripts/cnsl/April2026/exp_b_0_dry_run.sh
#   bash scripts/cnsl/April2026/exp_b_0_dry_run.sh -d 0

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

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE WANDB_MODE=offline \
  python src/train.py experiment=cnsl/April2026/exp_b_xlsr_conformertcm_mbct_mdt_optimized \
  +trainer.fast_dev_run=2 \
  trainer.accelerator=gpu trainer.devices=1 \
  data.batch_size=2 data.num_workers=0 \
  callbacks=default_loss_earlystop \
  ++test=False

echo "[exp_b_0_dry_run] OK — check logs for val/view_normal_* and train/loss."
