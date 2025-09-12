#!/bin/bash

# ICASSP26 Unified Benchmark Runner Script
# This script runs all AASIST-SSL and ConformerTCM benchmarks and calculates EER

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
RESET='\033[0m'

# Function to display colored text
print_color() {
    local color="$1"
    local text="$2"
    echo -e "${color}${text}${RESET}"
}

# Function to display usage
show_usage() {
    print_color "$BLUE" "┌─────────────────────────────────────────────────────────────────┐"
    print_color "$BLUE" "│                ICASSP26 Unified Benchmark Runner                │"
    print_color "$BLUE" "└─────────────────────────────────────────────────────────────────┘"
    echo ""
    print_color "$CYAN" "Usage: $0 -d <cuda_device> -r <results_folder> [-m <model_type>] [-g <group>] [-c <comment>]"
    echo ""
    print_color "$YELLOW" "Parameters:"
    echo "  -d <cuda_device>     CUDA device number (0, 1, 2, 3, ...)"
    echo "  -r <results_folder>  Results folder path"
    echo "  -m <model_type>      Model type: 'aasist', 'conformertcm', or 'all' (default: all)"
    echo "  -g <group>           Specific group to run: 'g1', 'g2', ..., 'g7', or 'all' (default: all)"
    echo "  -c <comment>         Comment for result identification (default: icassp26_benchmark)"
    echo ""
    print_color "$YELLOW" "Examples:"
    echo "  $0 -d 2 -r logs/results/icassp26                    # Run all models on GPU 2"
    echo "  $0 -d 2 -r logs/results/icassp26 -m aasist          # Run only AASIST models"
    echo "  $0 -d 2 -r logs/results/icassp26 -g g1              # Run only g1 (background_music_noise)"
    echo "  $0 -d 2 -r logs/results/icassp26 -m conformertcm -g g3  # Run only ConformerTCM g3"
    exit 1
}

# Print banner
print_banner() {
    clear
    print_color "$MAGENTA" "┌─────────────────────────────────────────────────────────────────┐"
    print_color "$MAGENTA" "│               🚀 ICASSP26 BENCHMARK RUNNER 🚀                   │"
    print_color "$MAGENTA" "└─────────────────────────────────────────────────────────────────┘"
    echo ""
}

# Parse command line arguments
CUDA_DEVICE=""
RESULTS_FOLDER=""
MODEL_TYPE="all"
GROUP="all"
COMMENT="icassp26_benchmark"

while getopts "d:r:m:g:c:h" opt; do
    case $opt in
        d) CUDA_DEVICE="$OPTARG" ;;
        r) RESULTS_FOLDER="$OPTARG" ;;
        m) MODEL_TYPE="$OPTARG" ;;
        g) GROUP="$OPTARG" ;;
        c) COMMENT="$OPTARG" ;;
        h) show_usage ;;
        *) show_usage ;;
    esac
done

# Check required arguments
if [ -z "$CUDA_DEVICE" ] || [ -z "$RESULTS_FOLDER" ]; then
    print_color "$RED" "Error: Missing required parameters"
    show_usage
fi

# Print banner
print_banner

# Validate model type
if [[ "$MODEL_TYPE" != "all" && "$MODEL_TYPE" != "aasist" && "$MODEL_TYPE" != "conformertcm" ]]; then
    print_color "$RED" "Error: Invalid model type. Must be 'all', 'aasist', or 'conformertcm'"
    exit 1
fi

# Validate group
if [[ "$GROUP" != "all" && ! "$GROUP" =~ ^g[1-7]$ ]]; then
    print_color "$RED" "Error: Invalid group. Must be 'all' or 'g1' through 'g7'"
    exit 1
fi

# Create results directory
mkdir -p "$RESULTS_FOLDER"

# Define model configurations
declare -A AASIST_CONFIGS=(
    ["g1"]="background_music_noise:/home/hungdx/logs/train/runs/2025-09-11_02-19-50/checkpoints/epoch_001.ckpt:/nvme1/hungdx/Lightning-hydra/protocols_icassp/background_music_noise.txt"
    ["g2"]="auto_tune:/home/hungdx/logs/train/runs/2025-09-11_02-21-04/checkpoints/epoch_018.ckpt:/nvme1/hungdx/Lightning-hydra/protocols_icassp/auto_tune.txt"
    ["g3"]="band_pass_filter:/home/hungdx/logs/train/runs/2025-09-11_04-04-11/checkpoints/epoch_029.ckpt:/nvme1/hungdx/Lightning-hydra/protocols_icassp/band_pass_filter.txt"
    ["g4"]="echo:/home/hungdx/logs/train/runs/2025-09-11_04-03-13/checkpoints/epoch_028.ckpt:/nvme1/hungdx/Lightning-hydra/protocols_icassp/echo.txt"
    ["g5"]="manipulation:/home/hungdx/logs/train/runs/2025-09-11_04-28-50/checkpoints/epoch_019.ckpt:/nvme1/hungdx/Lightning-hydra/protocols_icassp/manipulation.txt"
    ["g6"]="gaussian_noise:/home/hungdx/logs/train/runs/2025-09-11_04-47-15/checkpoints/epoch_029.ckpt:/nvme1/hungdx/Lightning-hydra/protocols_icassp/gaussian_noise.txt"
    ["g7"]="reverberation:/home/hungdx/logs/train/runs/2025-09-11_05-11-40/checkpoints/epoch_028.ckpt:/nvme1/hungdx/Lightning-hydra/protocols_icassp/reverberation.txt"
)

declare -A CONFORMERTCM_CONFIGS=(
    ["g1"]="background_music_noise:/home/hungdx/logs/train/runs/2025-09-11_05-38-35/checkpoints/epoch_008.ckpt:/nvme1/hungdx/Lightning-hydra/protocols_icassp/background_music_noise.txt"
    ["g2"]="auto_tune:/home/hungdx/logs/train/runs/2025-09-11_05-38-53/checkpoints/epoch_029.ckpt:/nvme1/hungdx/Lightning-hydra/protocols_icassp/auto_tune.txt"
    ["g3"]="band_pass_filter:/home/hungdx/logs/train/runs/2025-09-11_06-02-37/checkpoints/epoch_011.ckpt:/nvme1/hungdx/Lightning-hydra/protocols_icassp/band_pass_filter.txt"
    ["g4"]="echo:/home/hungdx/logs/train/runs/2025-09-11_06-07-41/checkpoints/epoch_029.ckpt:/nvme1/hungdx/Lightning-hydra/protocols_icassp/echo.txt"
    ["g5"]="manipulation:/home/hungdx/logs/train/runs/2025-09-11_06-29-53/checkpoints/epoch_017.ckpt:/nvme1/hungdx/Lightning-hydra/protocols_icassp/manipulation.txt"
    ["g6"]="gaussian_noise:/home/hungdx/logs/train/runs/2025-09-11_06-36-06/checkpoints/epoch_019.ckpt:/nvme1/hungdx/Lightning-hydra/protocols_icassp/gaussian_noise.txt"
    ["g7"]="reverberation:/home/hungdx/logs/train/runs/2025-09-11_07-05-42/checkpoints/epoch_021.ckpt:/nvme1/hungdx/Lightning-hydra/protocols_icassp/reverberation.txt"
)

# Function to run benchmark for a specific model and group
run_benchmark() {
    local model_name="$1"
    local group_id="$2"
    local config="$3"
    
    # Parse config: noise_type:checkpoint_path:protocol_path
    IFS=':' read -r noise_type checkpoint_path protocol_path <<< "$config"
    
    print_color "$YELLOW" "┌─────────────────────────────────────────────────────────────────┐"
    print_color "$YELLOW" "│ Running $model_name $group_id ($noise_type)"
    print_color "$YELLOW" "└─────────────────────────────────────────────────────────────────┘"
    
    # Check if checkpoint exists
    # if [ ! -f "$checkpoint_path" ]; then
    #     print_color "$RED" "❌ Warning: Checkpoint file not found: $checkpoint_path"
    #     print_color "$YELLOW" "Skipping $model_name $group_id"
    #     return 1
    # fi
    
    # Check if protocol exists
    if [ ! -f "$protocol_path" ]; then
        print_color "$RED" "❌ Warning: Protocol file not found: $protocol_path"
        print_color "$YELLOW" "Skipping $model_name $group_id"
        return 1
    fi
    
    # Set model-specific parameters
    local experiment_config=""
    local base_model_path=""
    local random_start="false"
    
    if [ "$model_name" == "aasist" ]; then
        experiment_config="icassp26/aasist_ssl/xlsr_aasist_single_lora"
        base_model_path="/nvme1/hungdx/Lightning-hydra/Best_LA_model_for_DF.pth"
        random_start="false"
    elif [ "$model_name" == "conformertcm" ]; then
        experiment_config="icassp26/conformertcm/xlsr_conformertcm_single_lora"
        base_model_path="/nvme1/hungdx/tcm_add/models/pretrained/DF/avg_5_best.pth"
        random_start="false"
    fi
    
    # Generate score save path
    local score_save_path="$RESULTS_FOLDER/${noise_type}_${model_name}_${group_id}_${COMMENT}.txt"
    
    # Construct and execute command
    local cmd="CUDA_VISIBLE_DEVICES=$CUDA_DEVICE OMP_NUM_THREADS=5 python src/train.py"
    cmd+=" experiment=$experiment_config"
    cmd+=" ++model.score_save_path=\"$score_save_path\""
    cmd+=" ++train=False ++test=True ++model.spec_eval=True"
    cmd+=" ++data.args.trim_length=64000"
    cmd+=" ++model.base_model_path=\"$base_model_path\""
    cmd+=" ++model.is_base_model_path_ln=false"
    cmd+=" ++model.adapter_paths=\"$checkpoint_path\""
    cmd+=" ++data.args.protocol_path=\"$protocol_path\""
    cmd+=" ++data.args.random_start=$random_start"
    
    print_color "$CYAN" "🔄 Executing benchmark..."
    print_color "$WHITE" "$cmd"
    echo ""
    
    # Execute the command
    eval $cmd
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_color "$GREEN" "✓ Benchmark completed successfully"
        
        # Calculate EER if score file exists
        if [ -f "$score_save_path" ]; then
            print_color "$CYAN" "🔄 Calculating EER..."
            local result=$(python scripts/score_file_to_eer.py "$score_save_path" "$protocol_path" 2>/dev/null)
            
            if [ $? -eq 0 ] && [ ! -z "$result" ]; then
                # Extract values from the result
                local min_score=$(echo "$result" | cut -d' ' -f1)
                local max_score=$(echo "$result" | cut -d' ' -f2)
                local threshold=$(echo "$result" | cut -d' ' -f3)
                local eer=$(echo "$result" | cut -d' ' -f4)
                local accuracy=$(echo "$result" | cut -d' ' -f5)
                
                # Display results
                print_color "$GREEN" "✓ Results for $model_name $group_id ($noise_type):"
                print_color "$WHITE" "  EER      : $eer"
                print_color "$WHITE" "  Accuracy : $accuracy"
                print_color "$WHITE" "  Threshold: $threshold"
                print_color "$WHITE" "  Min Score: $min_score"
                print_color "$WHITE" "  Max Score: $max_score"
                
                # Save to summary file
                echo "$model_name,$group_id,$noise_type,$eer,$accuracy,$threshold,$min_score,$max_score" >> "$RESULTS_FOLDER/summary_${COMMENT}.csv"
            else
                print_color "$RED" "❌ Failed to calculate EER"
            fi
        else
            print_color "$RED" "❌ Score file not generated"
        fi
    else
        print_color "$RED" "❌ Benchmark failed with exit code $exit_code"
    fi
    
    echo ""
    return $exit_code
}

# Function to get groups to process
get_groups_to_process() {
    if [ "$GROUP" == "all" ]; then
        echo "g1 g2 g3 g4 g5 g6 g7"
    else
        echo "$GROUP"
    fi
}

# Main execution
print_color "$CYAN" "🚀 Starting ICASSP26 Benchmark Runner"
print_color "$WHITE" "  CUDA Device: $CUDA_DEVICE"
print_color "$WHITE" "  Results Folder: $RESULTS_FOLDER"
print_color "$WHITE" "  Model Type: $MODEL_TYPE"
print_color "$WHITE" "  Group: $GROUP"
print_color "$WHITE" "  Comment: $COMMENT"
echo ""

# Initialize summary file
SUMMARY_FILE="$RESULTS_FOLDER/summary_${COMMENT}.csv"
echo "Model,Group,NoiseType,EER,Accuracy,Threshold,MinScore,MaxScore" > "$SUMMARY_FILE"

# Get groups to process
GROUPS_TO_PROCESS=$(get_groups_to_process)

# Initialize counters
TOTAL_BENCHMARKS=0
SUCCESSFUL_BENCHMARKS=0
FAILED_BENCHMARKS=0

# Count total benchmarks
for group in $GROUPS_TO_PROCESS; do
    if [[ "$MODEL_TYPE" == "all" || "$MODEL_TYPE" == "aasist" ]]; then
        TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))
    fi
    if [[ "$MODEL_TYPE" == "all" || "$MODEL_TYPE" == "conformertcm" ]]; then
        TOTAL_BENCHMARKS=$((TOTAL_BENCHMARKS + 1))
    fi
done

print_color "$CYAN" "📊 Total benchmarks to run: $TOTAL_BENCHMARKS"
echo ""

# Run AASIST benchmarks
if [[ "$MODEL_TYPE" == "all" || "$MODEL_TYPE" == "aasist" ]]; then
    print_color "$MAGENTA" "┌─────────────────────────────────────────────────────────────────┐"
    print_color "$MAGENTA" "│                    RUNNING AASIST BENCHMARKS                    │"
    print_color "$MAGENTA" "└─────────────────────────────────────────────────────────────────┘"
    
    for group in $GROUPS_TO_PROCESS; do
        if [ -n "${AASIST_CONFIGS[$group]}" ]; then
            if run_benchmark "aasist" "$group" "${AASIST_CONFIGS[$group]}"; then
                SUCCESSFUL_BENCHMARKS=$((SUCCESSFUL_BENCHMARKS + 1))
            else
                FAILED_BENCHMARKS=$((FAILED_BENCHMARKS + 1))
            fi
        else
            print_color "$RED" "❌ Unknown AASIST group: $group"
            FAILED_BENCHMARKS=$((FAILED_BENCHMARKS + 1))
        fi
    done
fi

# Run ConformerTCM benchmarks
if [[ "$MODEL_TYPE" == "all" || "$MODEL_TYPE" == "conformertcm" ]]; then
    print_color "$MAGENTA" "┌─────────────────────────────────────────────────────────────────┐"
    print_color "$MAGENTA" "│                 RUNNING CONFORMERTCM BENCHMARKS                 │"
    print_color "$MAGENTA" "└─────────────────────────────────────────────────────────────────┘"
    
    for group in $GROUPS_TO_PROCESS; do
        if [ -n "${CONFORMERTCM_CONFIGS[$group]}" ]; then
            if run_benchmark "conformertcm" "$group" "${CONFORMERTCM_CONFIGS[$group]}"; then
                SUCCESSFUL_BENCHMARKS=$((SUCCESSFUL_BENCHMARKS + 1))
            else
                FAILED_BENCHMARKS=$((FAILED_BENCHMARKS + 1))
            fi
        else
            print_color "$RED" "❌ Unknown ConformerTCM group: $group"
            FAILED_BENCHMARKS=$((FAILED_BENCHMARKS + 1))
        fi
    done
fi

# Final summary
print_color "$MAGENTA" "┌─────────────────────────────────────────────────────────────────┐"
print_color "$MAGENTA" "│                       BENCHMARK COMPLETE                        │"
print_color "$MAGENTA" "└─────────────────────────────────────────────────────────────────┘"

print_color "$GREEN" "✓ Successful benchmarks: $SUCCESSFUL_BENCHMARKS"
if [ $FAILED_BENCHMARKS -gt 0 ]; then
    print_color "$RED" "❌ Failed benchmarks: $FAILED_BENCHMARKS"
fi
print_color "$CYAN" "📊 Results summary: $SUMMARY_FILE"

# Display summary table if there are results
if [ -f "$SUMMARY_FILE" ] && [ $(wc -l < "$SUMMARY_FILE") -gt 1 ]; then
    echo ""
    print_color "$YELLOW" "📊 BENCHMARK RESULTS SUMMARY:"
    echo ""
    print_color "$WHITE" "$(cat "$SUMMARY_FILE" | column -t -s ',')"
fi

echo ""
print_color "$GREEN" "🎉 ICASSP26 Benchmark Runner completed!"
print_color "$CYAN" "📁 All results saved to: $RESULTS_FOLDER"
