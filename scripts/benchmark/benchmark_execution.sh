#!/bin/bash

# Benchmark Execution Module
# Handles benchmark command construction and execution

# Source utilities and constants
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/benchmark_utils.sh"
source "$SCRIPT_DIR/benchmark_constants.sh"

# Function to construct benchmark command
construct_benchmark_command() {
    local gpu_number="$1"
    local yaml_config="$2"
    local score_save_path="$3"
    local data_dir="$4"
    local protocol_path="$5"
    local base_model_path="$6"
    local is_base_model_path_ln="$7"
    local is_random_start="$8"
    local trim_length="$9"
    local adapter_paths="${10:-}"
    
    local cmd="CUDA_VISIBLE_DEVICES=\"$gpu_number\" python src/train.py experiment=$yaml_config "
    cmd+="++model.score_save_path=\"$score_save_path\" "
    cmd+="++data.data_dir=\"$data_dir\" "
    cmd+="++data.args.protocol_path=\"$protocol_path\" "
    cmd+="++train=False ++test=True ++model.spec_eval=True ++data.batch_size=$DEFAULT_BATCH_SIZE "
    cmd+="++data.args.random_start=$is_random_start "
    cmd+="++data.args.trim_length=$trim_length "
    cmd+="++model.base_model_path=\"$base_model_path\" "
    cmd+="++model.is_base_model_path_ln=$is_base_model_path_ln "
    
    # Add adapter paths if provided
    if [ ! -z "$adapter_paths" ]; then
        cmd+="++model.adapter_paths=\"$adapter_paths\" "
    fi
    
    echo "$cmd"
}

# Function to execute benchmark with spinner
execute_benchmark() {
    local cmd="$1"
    
    print_color "$CYAN" "🔄 Running benchmark..."
    print_color "$WHITE" "$cmd"
    echo ""
    
    # Execute the command with a spinner
    run_with_spinner "$cmd" "Benchmark"
    return $?
}

# Function to evaluate results and extract metrics
evaluate_results() {
    local score_file="$1"
    local protocol_file="$2"
    local summary_file="$3"
    local dataset_name="$4"
    
    print_color "$CYAN" "🔄 Evaluating results..."
    local result=$(python scripts/score_file_to_eer.py "$score_file" "$protocol_file")
    
    # Check if the evaluation script was successful
    if [ $? -eq 0 ]; then
        # Extract values from the result
        local min_score=$(echo "$result" | cut -d' ' -f1)
        local max_score=$(echo "$result" | cut -d' ' -f2)
        local threshold=$(echo "$result" | cut -d' ' -f3)
        local eer=$(echo "$result" | cut -d' ' -f4)
        local accuracy=$(echo "$result" | cut -d' ' -f5)
        
        # Format output for summary file
        echo "$dataset_name | $eer | $min_score | $max_score | $threshold | $accuracy" >> "$summary_file"
        
        # Display results
        print_color "$GREEN" "✓ Results for $dataset_name:"
        print_color "$WHITE" "  EER      : $eer"
        print_color "$WHITE" "  Accuracy : $accuracy"
        print_color "$WHITE" "  Threshold: $threshold"
        print_color "$WHITE" "  Min Score: $min_score"
        print_color "$WHITE" "  Max Score: $max_score"
        
        return 0
    else
        print_color "$RED" "❌ Error: Failed to evaluate results for $dataset_name"
        return 1
    fi
}
