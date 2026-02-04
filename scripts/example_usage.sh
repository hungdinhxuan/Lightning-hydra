#!/bin/bash

# Example usage of the updated main_generate_ssbolldata_parallel.py script
# Parent directory should contain multiple dataset subdirectories, each with protocol.txt and audio files
# Only processes 'eval' subset entries from each protocol.txt
# Output maintains the exact same structure as input protocol

# Process all datasets in a parent directory (eval subset only)
python main_generate_ssbolldata_parallel.py \
    --parent_dir "/data/Datasets" \
    --target_base_dir "/nvme1/hungdx/Lightning-hydra/data/processed_ssboll79" \
    --post_name "_ssboll79" \
    --num_processes 8 \
    --IS 0.25

# Example with wildspoof challenge benchmark datasets (eval subset only)
python main_generate_ssbolldata_parallel.py \
    --parent_dir "/nvme1/hungdx/Lightning-hydra/data/wildspoof_challenge_benchmark" \
    --target_base_dir "/nvme1/hungdx/Lightning-hydra/data/processed_ssboll79" \
    --post_name "_ssboll79" \
    --num_processes 16 \
    --IS 0.25

# Example with custom parameters (eval subset only)
python main_generate_ssbolldata_parallel.py \
    --parent_dir "/path/to/your/datasets" \
    --target_base_dir "/path/to/output" \
    --post_name "_denoised" \
    --num_processes 32 \
    --IS 0.5
