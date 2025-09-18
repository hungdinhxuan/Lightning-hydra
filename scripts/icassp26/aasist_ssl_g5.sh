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
# CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=1 python src/train.py \
#     experiment=icassp26/aasist_ssl/xlsr_aasist_single_lora \
#     ++model.is_base_model_path_ln=false \
#     ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/protocols_icassp/manipulation.txt" \
#     logger=wandb
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=1 python src/train.py \
    experiment=icassp26/aasist_ssl/xlsr_aasist_single_lora \
    ++model.is_base_model_path_ln=false \
    ++data.args.protocol_path="/nvme1/hungdx/Lightning-hydra/noise_type_large_asv19/manipulation.txt" \
    ++data.data_dir="/nvme1/hungdx/Lightning-hydra/data/ASV19_noise" \
    ++data.batch_size=16 \
    ++model.optimizer.lr=0.0001 \
    logger=wandb