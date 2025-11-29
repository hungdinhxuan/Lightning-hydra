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


# This script is use pre-trained xlsr-conformertcm model and new initialized BEATs model to train the auxiliary task.

# Run the training command
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=5 python src/train.py \
    experiment=wildspoof/Nov3/xlsr_conformertcm \
    ++data.data_dir="data/WildSpoof/spoofceleb_aug" \
    ++data.args.protocol_path="data/WildSpoof/spoofceleb_aug/combined_noisy_vocoded_protocol.txt" logger=wandb \
    ++model.is_base_model_path_ln=True \
    ++model.base_model_path="pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt" \
    ++model_averaging=True