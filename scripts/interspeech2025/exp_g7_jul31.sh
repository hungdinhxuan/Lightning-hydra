#!/bin/bash

# Parse command line arguments
while getopts "d:" opt; do
  case $opt in
    d) CUDA_DEVICE="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
  esac
done

# Default CUDA device if not specified
CUDA_DEVICE=${CUDA_DEVICE:-3}

# Run the training command
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=1 python src/train.py \
    experiment=cnsl/lora/interspeech2025/Jul31/xlsr_conformertcm_lora_group7 \
    ++model.is_base_model_path_ln=false \
    logger=wandb 


# logger=wandb \