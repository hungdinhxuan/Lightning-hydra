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
    experiment=cnsl/lora/ft/July2/test \
    ++model.is_base_model_path_ln=false \
    logger=csv \
    ++data.data_dir="/nvme1/hungdx/Lightning-hydra/data/normal_250625_noise" \
    ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/data/normal_250625_noise/protocols/replay_protocol.txt" \
    +trainer.limit_train_batches=0.2 trainer.max_epochs=2


# logger=wandb \