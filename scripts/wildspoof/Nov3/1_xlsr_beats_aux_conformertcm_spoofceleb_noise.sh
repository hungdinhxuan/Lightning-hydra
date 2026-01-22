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


# This script is use pre-trained xlsr-conformertcm model and new initialized BEATs model to train the auxiliary task.

# Run the training command
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=5 python src/train.py \
    experiment=wildspoof/Nov3/xlsr_beats_conformertcm_aux_fixed \
    ++data.data_dir="/data/Datasets/spoofceleb_spoof_noise_dataset" \
    ++data.args.protocol_path="data/Wild_Spoof_Dataset/spoofceleb_noise_dataset/protocol_aux.txt" logger=wandb \
    ++model.base_model_path="pretrained/xlsr_conformertcm_mdt_rawboost3_spoofceleb.pt" \
    ++model.is_base_model_path_ln=True \
    ++model_averaging=True