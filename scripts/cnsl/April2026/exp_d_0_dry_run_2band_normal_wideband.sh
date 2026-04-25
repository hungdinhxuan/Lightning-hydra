#!/bin/bash
# EXP D dry run: MDT×MBCT LoRA — two bands (normal + wideband).
# experiment=cnsl/April2026/exp_d_xlsr_conformertcm_mbct_mdt_lora_2band_normal_wideband
#
# Usage:
#   export XLSR_PRETRAINED_MODEL_PATH=pretrained/xlsr2_300m.pt
#   bash scripts/cnsl/April2026/exp_d_0_dry_run_2band_normal_wideband.sh
#   bash scripts/cnsl/April2026/exp_d_0_dry_run_2band_normal_wideband.sh -d 0

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
  python src/train.py \
  experiment=cnsl/April2026/exp_d_xlsr_conformertcm_mbct_mdt_lora_2band_normal_wideband \
  +trainer.fast_dev_run=2 \
  trainer.accelerator=gpu trainer.devices=1 \
  data.batch_size=2 data.num_workers=0 \
  callbacks=default_lora_loss_earlystop \
  ++data.data_dir=data/DVC_DSD-Large-Corpus/raw/0_large-corpus_toys \
  ++data.args.protocol_path=data/protocols/cnsl/new_protocol_trim_vocoded_cleaned_v4_corrected.txt \
  ++test=False

echo "[exp_d_0_dry_run_2band_normal_wideband] OK — check logs for val/view_* and train/loss."
