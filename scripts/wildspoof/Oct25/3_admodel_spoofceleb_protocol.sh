#!/bin/bash

# Parse command line arguments
while getopts "d:" opt; do
  case $opt in
    d) CUDA_DEVICE="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
  esac
done

# Default CUDA device if not specified
CUDA_DEVICE=${CUDA_DEVICE:-1}

# Run the training command
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=5 python src/train.py \
    experiment=wildspoof/Oct_25/xlsr_seres2net_conformertcm_mdt \
    ++data.args.protocol_path="data/Wild_Spoof_Dataset/spoofceleb/protocol.txt" \
    ++data.data_dir="data/Wild_Spoof_Dataset/spoofceleb" \
    +trainer.limit_train_batches=0.01 logger=wandb # 