#!/bin/bash

# Parse command line arguments
while getopts "d:" opt; do
  case $opt in
    d) CUDA_DEVICE="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
  esac
done

# Default CUDA device if not specified (MIG GPU hash or device ID)
CUDA_DEVICE=${CUDA_DEVICE:-"MIG-57de94a5-be15-5b5a-b67e-e118352d8a59"}


# This script is use pre-trained xlsr-conformertcm model and apply VAD as preprocesing for clean silence from dataset.

# Run the training command
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=5 uv run src/train.py \
    experiment=cnsl/March2026/xlsr_conformertcm_mdt \
    ++data.data_dir="data/replay_cl_260320" \
    ++data.args.protocol_path="data/replay_cl_260320/protocols/protocol.txt" logger=wandb \
    +trainer.val_check_interval=0.25 \
    ++model_averaging=True
