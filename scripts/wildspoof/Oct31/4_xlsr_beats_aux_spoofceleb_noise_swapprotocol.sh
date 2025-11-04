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
    experiment=wildspoof/Oct31/xlsr_beats_seres2net_conformertcm_aux \
    ++data.data_dir="/data/Datasets/spoofceleb_spoof_noise_dataset" \
    ++data.args.protocol_path="data/Wild_Spoof_Dataset/spoofceleb_noise_dataset/protocol_aux2.txt" logger=wandb ++model_averaging=True