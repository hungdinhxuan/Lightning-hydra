#!/bin/bash

# Test if the benchmark command actually works
cd /nvme1/hungdx/code/Lightning-hydra

CUDA_VISIBLE_DEVICES="MIG-6e4275af-2db0-51f1-a601-7ad8a1002745" python src/train.py \
  experiment=cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer \
  ++model.score_save_path=/nvme1/hungdx/code/Lightning-hydra/logs/results/test_output.txt \
  ++data.data_dir=/nvme1/hungdx/code/Lightning-hydra/data/CNSL_Q1_2026_benchmarks/2025_April \
  ++data.args.protocol_path=/nvme1/hungdx/code/Lightning-hydra/logs/results/CNSL_Q1_2026_benchmarks/XLSR_ConformerTCM_FT_TelephonyLA/temp_protocol_2025_April_1152366_1769401427_16107.txt \
  ++train=False \
  ++test=True \
  ++model.spec_eval=True \
  ++data.batch_size=128 \
  ++data.args.random_start=true \
  ++data.args.trim_length=64000 \
  ++model.base_model_path=/home/hungdx/code/Lightning-hydra/logs/train/runs/2026-01-25_02-22-18/checkpoints/averaged_top5.ckpt \
  ++model.is_base_model_path_ln=true

echo "Exit code: $?"
echo "Output file created:"
ls -lh /nvme1/hungdx/code/Lightning-hydra/logs/results/test_output.txt 2>&1
