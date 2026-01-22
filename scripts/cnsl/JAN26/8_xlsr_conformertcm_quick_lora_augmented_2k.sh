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
    experiment=cnsl/Jan2026/xlsr_conformertcm_quick_lora_augmented \
    ++data.data_dir="data/DVC_DSD-Large-Corpus/pool/dsd_corpus_may_training_la_codec_aug_2k" \
    ++data.args.protocol_path="data/DVC_DSD-Large-Corpus/pool/dsd_corpus_may_training_la_codec_aug_2k/protocol.txt" logger=wandb \
    ++model.is_base_model_path_ln=False \
    ++model.base_model_path="pretrained/S_241214_conf-1.pth" \
    ++data.args.enable_cache=false