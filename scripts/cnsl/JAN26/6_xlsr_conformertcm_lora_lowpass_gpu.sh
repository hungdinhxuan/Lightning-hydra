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
    experiment=cnsl/Jan2026/xlsr_conformertcm_mdt_lora_lowpass_gpu_aug \
    ++data.data_dir="data/DVC_DSD-Large-Corpus/raw/0_large-corpus_toys" \
    ++data.args.protocol_path="data/protocols/cnsl/new_protocol_trim_vocoded_cleaned_v4_corrected.txt" logger=wandb \
    ++model.is_base_model_path_ln=False \
    ++model.base_model_path="pretrained/S_241214_conf-1.pth" \
    ++data.args.enable_cache=false