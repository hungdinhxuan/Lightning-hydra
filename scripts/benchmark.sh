#!/bin/bash

# Bulk Benchmark Runner Script with Colors and Progress Tracking

# Colors
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

# Function to display a progress bar
display_progress() {
    local current="$1"
    local total="$2"
    
    # Prevent division by zero
    if [ "$total" -eq 0 ]; then
        print_color "$RED" "Error: No subdirectories found to process."
        return 1
    fi
    
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((width * current / total))
    local remaining=$((width - completed))
    
    printf "${WHITE}[${GREEN}"
    for ((i=0; i<completed; i++)); do
        printf "="
    done
    
    if [[ $completed -lt $width ]]; then
        printf ">"
        for ((i=0; i<remaining-1; i++)); do
            printf " "
        done
    fi
    
    printf "${WHITE}] ${percentage}%% (${current}/${total})${RESET}\n"
}

# Function to display usage information
show_usage() {
    print_color "$BLUE" "┌─────────────────────────────────────────────────────────────────┐"
    print_color "$BLUE" "│                 Bulk Benchmark Runner Script                    │"
    print_color "$BLUE" "└─────────────────────────────────────────────────────────────────┘"
    echo ""
    print_color "$CYAN" "Usage: $0 -g <gpu_number> -c <yaml_config_file> -b <bulk_benchmark_folder> -m <base_model_path> -r <results_folder> -n <comment> [-a <adapter_paths>] [-l <is_base_model_path_ln>]"
    echo ""
    print_color "$YELLOW" "Parameters:"
    echo "  -g <gpu_number>             GPU number to use (0, 1, 2, 3, ...)"
    echo "  -c <yaml_config_file>       Yaml config file path (e.g., cnsl/xlsr_vib_large_corpus)"
    echo "  -b <bulk_benchmark_folder>  Bulk benchmark folder path"
    echo "  -m <base_model_path>        Base model path"
    echo "  -r <results_folder>         Results folder path"
    echo "  -n <comment>                Comment to note"
    echo "  -a <adapter_paths>          Adapter paths (optional)"
    echo "  -l <is_base_model_path_ln>  Whether to use Lightning checkpoint loading (default: true)"
    exit 1
}

# Function to print banner
print_banner() {
    clear
    print_color "$MAGENTA" "┌─────────────────────────────────────────────────────────────────┐"
    print_color "$MAGENTA" "│               🚀 BULK BENCHMARK RUNNER TOOL 🚀                  │"
    print_color "$MAGENTA" "└─────────────────────────────────────────────────────────────────┘"
    echo ""
}

# Print banner
print_banner

# Parse command line arguments
while getopts "g:c:b:m:r:n:a:l:" opt; do
    case $opt in
        g) GPU_NUMBER="$OPTARG" ;;
        c) YAML_CONFIG="$OPTARG" ;;
        b) BENCHMARK_FOLDER="$OPTARG" ;;
        m) BASE_MODEL_PATH="$OPTARG" ;;
        r) RESULTS_FOLDER="$OPTARG" ;;
        n) COMMENT="$OPTARG" ;;
        a) ADAPTER_PATHS="$OPTARG" ;;
        l) IS_BASE_MODEL_PATH_LN="$OPTARG" ;;
        *) show_usage ;;
    esac
done

# Check if required arguments are provided
if [ -z "$GPU_NUMBER" ] || [ -z "$YAML_CONFIG" ] || [ -z "$BENCHMARK_FOLDER" ] || [ -z "$BASE_MODEL_PATH" ] || [ -z "$RESULTS_FOLDER" ]; then
    print_color "$RED" "Error: Missing required parameters"
    show_usage
fi

# Set default value for IS_BASE_MODEL_PATH_LN if not provided
if [ -z "$IS_BASE_MODEL_PATH_LN" ]; then
    IS_BASE_MODEL_PATH_LN="true"
fi

# Ensure benchmark folder exists
if [ ! -d "$BENCHMARK_FOLDER" ]; then
    print_color "$RED" "Error: Benchmark folder '$BENCHMARK_FOLDER' does not exist."
    exit 1
fi

# Create complete results directory with comment subfolder
RESULTS_FOLDER="${RESULTS_FOLDER%/}/${COMMENT}"
mkdir -p "$RESULTS_FOLDER"

# Create a summary file
SUMMARY_FILE="$RESULTS_FOLDER/summary_results.txt"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

# Normalize YAML config for file naming
NORMALIZED_YAML=$(echo "$YAML_CONFIG" | tr '/' '_')

# Write header to summary file
echo "Config: $YAML_CONFIG" > "$SUMMARY_FILE"
echo "Base_model_path: $BASE_MODEL_PATH" >> "$SUMMARY_FILE"
echo "Lora Path: ${ADAPTER_PATHS:-None}" >> "$SUMMARY_FILE"
echo "Is Base Model Path LN: $IS_BASE_MODEL_PATH_LN" >> "$SUMMARY_FILE"
echo "Date: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Dataset | EER | min_score | max_score | Threshold | Accuracy" >> "$SUMMARY_FILE"

# Get list of subdirectories using ls instead of find
print_color "$CYAN" "Checking subdirectories in '$BENCHMARK_FOLDER'..."
SUBDIRS=()
while IFS= read -r dir; do
    if [[ -d "$BENCHMARK_FOLDER/$dir" ]]; then
        print_color "$WHITE" "Found directory: $dir"
        SUBDIRS+=("$BENCHMARK_FOLDER/$dir")
    fi
done < <(ls -1 "$BENCHMARK_FOLDER")

# Count total number of subfolders for progress tracking
TOTAL_SUBFOLDERS=${#SUBDIRS[@]}

if [ $TOTAL_SUBFOLDERS -eq 0 ]; then
    print_color "$RED" "Error: No subdirectories found in '$BENCHMARK_FOLDER'."
    # Debug output to help diagnose the issue
    print_color "$YELLOW" "Directory contents:"
    ls -la "$BENCHMARK_FOLDER"
    exit 1
fi

print_color "$GREEN" "✓ Found $TOTAL_SUBFOLDERS datasets to process"
print_color "$CYAN" "✓ Starting benchmark with GPU $GPU_NUMBER and config $YAML_CONFIG"
print_color "$CYAN" "✓ Results will be saved to $RESULTS_FOLDER"
echo ""

# Process each subfolder in the benchmark folder
CURRENT_SUBFOLDER=0
for subfolder in "${SUBDIRS[@]}"; do
    CURRENT_SUBFOLDER=$((CURRENT_SUBFOLDER + 1))
    subfolder_name=$(basename "$subfolder")
    
    # Display progress
    print_color "$YELLOW" "┌─────────────────────────────────────────────────────────────────┐"
    print_color "$YELLOW" "│ Processing dataset: $subfolder_name"
    print_color "$YELLOW" "└─────────────────────────────────────────────────────────────────┘"
    display_progress $CURRENT_SUBFOLDER $TOTAL_SUBFOLDERS
    
    # Set paths
    DATA_DIR="$subfolder"
    PROTOCOL_PATH="$subfolder/protocol.txt"
    SCORE_SAVE_PATH="$RESULTS_FOLDER/${subfolder_name}_${NORMALIZED_YAML}_${COMMENT}.txt"

    if [ -f "$SCORE_SAVE_PATH" ]; then
        # Run the scoring script
        print_color "$CYAN" "🔄 Evaluating results..."
        RESULT=$(python scripts/score_file_to_eer.py "$SCORE_SAVE_PATH" "$PROTOCOL_PATH")
        
        # Check if the evaluation script was successful
        if [ $? -eq 0 ]; then
            # Extract values from the result
            MIN_SCORE=$(echo "$RESULT" | cut -d' ' -f1)
            MAX_SCORE=$(echo "$RESULT" | cut -d' ' -f2)
            THRESHOLD=$(echo "$RESULT" | cut -d' ' -f3)
            EER=$(echo "$RESULT" | cut -d' ' -f4)
            ACCURACY=$(echo "$RESULT" | cut -d' ' -f5)
            
            # Format output for summary file
            echo "$subfolder_name | $EER | $MIN_SCORE | $MAX_SCORE | $THRESHOLD | $ACCURACY" >> "$SUMMARY_FILE"
            
            # Display results
            print_color "$GREEN" "✓ Results for $subfolder_name:"
            print_color "$WHITE" "  EER      : $EER"
            print_color "$WHITE" "  Accuracy : $ACCURACY"
            print_color "$WHITE" "  Threshold: $THRESHOLD"
            print_color "$WHITE" "  Min Score: $MIN_SCORE"
            print_color "$WHITE" "  Max Score: $MAX_SCORE"
            continue
        else
            print_color "$RED" "❌ Error: Failed to evaluate results for $subfolder_name"
        fi
    else
        print_color "$RED" "❌ Error: Score file was not created for $subfolder_name"
    fi
    
    # Check if protocol file exists
    if [ ! -f "$PROTOCOL_PATH" ]; then
        print_color "$RED" "⚠️ Warning: Protocol file not found at $PROTOCOL_PATH. Skipping this dataset."
        continue
    fi
    
    # Construct command
    CMD="CUDA_VISIBLE_DEVICES=$GPU_NUMBER python src/train.py experiment=$YAML_CONFIG "
    CMD+="++model.score_save_path=\"$SCORE_SAVE_PATH\" "
    CMD+="++data.data_dir=\"$DATA_DIR\" "
    CMD+="++data.args.protocol_path=\"$PROTOCOL_PATH\" "
    CMD+="++train=False ++test=True ++model.spec_eval=True ++data.batch_size=128 "
    CMD+="++model.base_model_path=\"$BASE_MODEL_PATH\" "
    CMD+="++model.is_base_model_path_ln=$IS_BASE_MODEL_PATH_LN "
    
    # Add adapter paths if provided
    if [ ! -z "$ADAPTER_PATHS" ]; then
        CMD+="++model.adapter_paths=\"$ADAPTER_PATHS\" "
    fi
    
    # Execute the command
    print_color "$CYAN" "🔄 Running benchmark..."
    print_color "$WHITE" "$CMD"
    echo ""
    
    # Execute the command with a spinner
    eval $CMD &
    PID=$!
    
    # Display a spinner while the command is running
    spin='-\|/'
    i=0
    while kill -0 $PID 2>/dev/null; do
        i=$(( (i+1) % 4 ))
        printf "\r${CYAN}⏳ Processing: %c${RESET}" "${spin:$i:1}"
        sleep .1
    done
    printf "\r${GREEN}✓ Benchmark completed                 ${RESET}\n"
    
    # Check if the score file was created
    if [ -f "$SCORE_SAVE_PATH" ]; then
        # Run the scoring script
        print_color "$CYAN" "🔄 Evaluating results..."
        RESULT=$(python scripts/score_file_to_eer.py "$SCORE_SAVE_PATH" "$PROTOCOL_PATH")
        
        # Check if the evaluation script was successful
        if [ $? -eq 0 ]; then
            # Extract values from the result
            MIN_SCORE=$(echo "$RESULT" | cut -d' ' -f1)
            MAX_SCORE=$(echo "$RESULT" | cut -d' ' -f2)
            THRESHOLD=$(echo "$RESULT" | cut -d' ' -f3)
            EER=$(echo "$RESULT" | cut -d' ' -f4)
            ACCURACY=$(echo "$RESULT" | cut -d' ' -f5)
            
            # Format output for summary file
            echo "$subfolder_name | $EER | $MIN_SCORE | $MAX_SCORE | $THRESHOLD | $ACCURACY" >> "$SUMMARY_FILE"
            
            # Display results
            print_color "$GREEN" "✓ Results for $subfolder_name:"
            print_color "$WHITE" "  EER      : $EER"
            print_color "$WHITE" "  Accuracy : $ACCURACY"
            print_color "$WHITE" "  Threshold: $THRESHOLD"
            print_color "$WHITE" "  Min Score: $MIN_SCORE"
            print_color "$WHITE" "  Max Score: $MAX_SCORE"
        else
            print_color "$RED" "❌ Error: Failed to evaluate results for $subfolder_name"
        fi
    else
        print_color "$RED" "❌ Error: Score file was not created for $subfolder_name"
    fi
    
    print_color "$GREEN" "✓ Finished processing $subfolder_name"
    echo ""
done

# Final summary
print_color "$MAGENTA" "┌─────────────────────────────────────────────────────────────────┐"
print_color "$MAGENTA" "│                       BENCHMARK COMPLETE                        │"
print_color "$MAGENTA" "└─────────────────────────────────────────────────────────────────┘"
print_color "$GREEN" "✓ All benchmarks completed successfully!"
print_color "$CYAN" "✓ Summary available at: $SUMMARY_FILE"

# Pretty print the summary file
echo ""
print_color "$YELLOW" "📊 SUMMARY OF RESULTS:"
echo ""
print_color "$WHITE" "$(cat $SUMMARY_FILE | sed 's/|/│/g')"
echo ""
print_color "$GREEN" "Thanks for using the Bulk Benchmark Runner Tool!"