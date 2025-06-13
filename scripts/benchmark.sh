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
    
    # Initialize variables for protocol and score path handling
    PROTOCOL_TO_USE="$PROTOCOL_PATH"
    SCORE_PATH_TO_USE="$SCORE_SAVE_PATH"
    USE_TEMP_PROTOCOL=false
    TEMP_PROTOCOL_PATH=""
    TEMP_SCORE_PATH=""
    RANDOM_ID=""

    # Function to validate score file completeness
    validate_score_file() {
        local score_file="$1"
        local protocol_file="$2"
        
        if [ ! -f "$score_file" ]; then
            return 1  # Score file doesn't exist
        fi
        
        # Count lines in score file (excluding empty lines)
        local score_lines=$(grep -c "^[^[:space:]]*[[:space:]]" "$score_file" 2>/dev/null || echo "0")
        
        # Count evaluation lines in protocol file (assuming 'eval' subset, adjust if different)
        local eval_lines=$(grep -c "eval" "$protocol_file" 2>/dev/null || echo "0")
        
        # If no 'eval' subset found, try counting all lines (fallback)
        if [ "$eval_lines" -eq 0 ]; then
            eval_lines=$(grep -c "^[^[:space:]]*[[:space:]]" "$protocol_file" 2>/dev/null || echo "0")
        fi
        
        print_color "$WHITE" "  Score file lines: $score_lines"
        print_color "$WHITE" "  Expected lines (eval subset): $eval_lines"
        
        if [ "$score_lines" -eq "$eval_lines" ] && [ "$score_lines" -gt 0 ]; then
            return 0  # Valid and complete
        else
            return 2  # Incomplete or corrupted
        fi
    }

    # Function to create temporary protocol file with missing entries (optimized for sequential evaluation)
    create_missing_protocol() {
        local score_file="$1"
        local protocol_file="$2"
        local temp_protocol="$3"
        
        print_color "$CYAN" "🔍 Analyzing missing entries (optimized for sequential evaluation)..."
        
        # Count existing score lines
        local existing_lines=0
        if [ -f "$score_file" ]; then
            existing_lines=$(grep -c "^[^[:space:]]*[[:space:]]" "$score_file" 2>/dev/null || echo "0")
        fi
        
        print_color "$WHITE" "  Existing score lines: $existing_lines"
        
        # Create temporary file for eval subset
        local temp_id="$$_$(date +%s)_$RANDOM"
        local temp_protocol_eval="/tmp/protocol_eval_${temp_id}.txt"
        
        # Extract eval entries from protocol file
        if grep -q "eval" "$protocol_file"; then
            # If protocol has eval subset, use only eval lines
            grep "eval" "$protocol_file" > "$temp_protocol_eval"
        else
            # If no eval subset, use all lines
            cp "$protocol_file" "$temp_protocol_eval"
        fi
        
        local total_eval_lines=$(wc -l < "$temp_protocol_eval")
        print_color "$WHITE" "  Total eval lines in protocol: $total_eval_lines"
        
        # Calculate missing lines (sequential evaluation - just skip processed lines)
        local missing_count=$((total_eval_lines - existing_lines))
        
        if [ $missing_count -le 0 ]; then
            print_color "$GREEN" "  No missing entries found."
            rm -f "$temp_protocol_eval"
            touch "$temp_protocol"  # Create empty temp protocol
            return 0
        fi
        
        # Create temporary protocol file with remaining entries (starting from existing_lines + 1)
        local start_line=$((existing_lines + 1))
        tail -n +$start_line "$temp_protocol_eval" > "$temp_protocol"
        
        print_color "$YELLOW" "  Found $missing_count missing entries (starting from line $start_line)"
        
        # Clean up temporary files
        rm -f "$temp_protocol_eval"
        
        return $missing_count
    }

    # Function to merge score files
    merge_score_files() {
        local original_score="$1"
        local new_score="$2"
        local merged_score="$3"
        
        print_color "$CYAN" "🔄 Merging score files..."
        
        # Create backup of original score file
        if [ -f "$original_score" ]; then
            cp "$original_score" "${original_score}.backup"
        fi
        
        # Combine original and new scores, then sort by first column
        if [ -f "$original_score" ] && [ -f "$new_score" ]; then
            cat "$original_score" "$new_score" | sort -k1,1 > "$merged_score"
        elif [ -f "$new_score" ]; then
            cp "$new_score" "$merged_score"
        elif [ -f "$original_score" ]; then
            cp "$original_score" "$merged_score"
        fi
        
        # Replace original with merged
        if [ -f "$merged_score" ]; then
            mv "$merged_score" "$original_score"
            print_color "$GREEN" "✓ Score files merged successfully"
        fi
    }
    
    # Check if score file exists and is complete
    if validate_score_file "$SCORE_SAVE_PATH" "$PROTOCOL_PATH"; then
        # Run the scoring script
        print_color "$CYAN" "🔄 Evaluating existing complete results..."
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
            print_color "$GREEN" "✓ Results for $subfolder_name (using existing complete score file):"
            print_color "$WHITE" "  EER      : $EER"
            print_color "$WHITE" "  Accuracy : $ACCURACY"
            print_color "$WHITE" "  Threshold: $THRESHOLD"
            print_color "$WHITE" "  Min Score: $MIN_SCORE"
            print_color "$WHITE" "  Max Score: $MAX_SCORE"
            continue
        else
            print_color "$RED" "❌ Error: Failed to evaluate results for $subfolder_name"
        fi
    elif [ -f "$SCORE_SAVE_PATH" ]; then
        print_color "$YELLOW" "⚠️ Warning: Score file exists but is incomplete/corrupted for $subfolder_name."
        
        # Create temporary protocol file with missing entries (using random ID to avoid conflicts)
        RANDOM_ID="$$_$(date +%s)_$RANDOM"
        TEMP_PROTOCOL_PATH="$RESULTS_FOLDER/temp_protocol_${subfolder_name}_${RANDOM_ID}.txt"
        
        create_missing_protocol "$SCORE_SAVE_PATH" "$PROTOCOL_PATH" "$TEMP_PROTOCOL_PATH"
        missing_count=$?
        
        if [ $missing_count -gt 0 ]; then
            print_color "$CYAN" "🔄 Running benchmark for $missing_count missing entries only..."
            PROTOCOL_TO_USE="$TEMP_PROTOCOL_PATH"
            TEMP_SCORE_PATH="$RESULTS_FOLDER/temp_scores_${subfolder_name}_${RANDOM_ID}.txt"
            SCORE_PATH_TO_USE="$TEMP_SCORE_PATH"
            USE_TEMP_PROTOCOL=true
        elif [ $missing_count -eq 0 ]; then
            print_color "$GREEN" "✓ No missing entries found. Score file is actually complete."
            # Re-validate to make sure
            if validate_score_file "$SCORE_SAVE_PATH" "$PROTOCOL_PATH"; then
                # Run the scoring script directly since file is complete
                print_color "$CYAN" "🔄 Evaluating existing complete results..."
                RESULT=$(python scripts/score_file_to_eer.py "$SCORE_SAVE_PATH" "$PROTOCOL_PATH")
                
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
                    print_color "$GREEN" "✓ Results for $subfolder_name (using existing complete score file):"
                    print_color "$WHITE" "  EER      : $EER"
                    print_color "$WHITE" "  Accuracy : $ACCURACY"
                    print_color "$WHITE" "  Threshold: $THRESHOLD"
                    print_color "$WHITE" "  Min Score: $MIN_SCORE"
                    print_color "$WHITE" "  Max Score: $MAX_SCORE"
                fi
            fi
            continue
        else
            print_color "$RED" "❌ Error: Failed to analyze missing entries. Re-running full benchmark..."
            PROTOCOL_TO_USE="$PROTOCOL_PATH"
            SCORE_PATH_TO_USE="$SCORE_SAVE_PATH"
            USE_TEMP_PROTOCOL=false
        fi
    else
        print_color "$CYAN" "ℹ️ No existing score file found for $subfolder_name. Running fresh benchmark..."
        PROTOCOL_TO_USE="$PROTOCOL_PATH"
        SCORE_PATH_TO_USE="$SCORE_SAVE_PATH"
        USE_TEMP_PROTOCOL=false
    fi
    
    # Check if protocol file exists
    if [ ! -f "$PROTOCOL_TO_USE" ]; then
        print_color "$RED" "⚠️ Warning: Protocol file not found at $PROTOCOL_TO_USE. Skipping this dataset."
        continue
    fi
    
    # Construct command
    CMD="CUDA_VISIBLE_DEVICES=$GPU_NUMBER python src/train.py experiment=$YAML_CONFIG "
    CMD+="++model.score_save_path=\"$SCORE_PATH_TO_USE\" "
    CMD+="++data.data_dir=\"$DATA_DIR\" "
    CMD+="++data.args.protocol_path=\"$PROTOCOL_TO_USE\" "
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
    
    # Handle temporary protocol case - merge results if needed
    if [ "$USE_TEMP_PROTOCOL" = true ]; then
        if [ -f "$TEMP_SCORE_PATH" ]; then
            print_color "$CYAN" "🔄 Merging temporary scores with existing scores..."
            MERGED_SCORE_PATH="$RESULTS_FOLDER/merged_scores_${subfolder_name}_${RANDOM_ID}.txt"
            merge_score_files "$SCORE_SAVE_PATH" "$TEMP_SCORE_PATH" "$MERGED_SCORE_PATH"
            
            # Clean up temporary files
            rm -f "$TEMP_PROTOCOL_PATH" "$TEMP_SCORE_PATH" "$MERGED_SCORE_PATH"
            print_color "$GREEN" "✓ Temporary files cleaned up"
        else
            print_color "$RED" "❌ Error: Temporary score file was not created for $subfolder_name"
        fi
    fi
    
    # Check if the final score file is complete
    if validate_score_file "$SCORE_SAVE_PATH" "$PROTOCOL_PATH"; then
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
    elif [ -f "$SCORE_SAVE_PATH" ]; then
        print_color "$RED" "❌ Error: Score file exists but is incomplete/corrupted for $subfolder_name"
    else
        print_color "$RED" "❌ Error: Score file was not created for $subfolder_name"
    fi
    
    print_color "$GREEN" "✓ Finished processing $subfolder_name"
    echo ""
done

# Calculate pooled EER from all datasets using combined temporary files
print_color "$MAGENTA" "┌─────────────────────────────────────────────────────────────────┐"
print_color "$MAGENTA" "│                    CALCULATING POOLED EER                       │"
print_color "$MAGENTA" "└─────────────────────────────────────────────────────────────────┘"

# Function to calculate pooled EER using dedicated Python script
calculate_pooled_eer() {
    print_color "$CYAN" "🔄 Calculating pooled EER using efficient Python implementation..."
    
    # Build command array with proper argument passing
    local cmd_args=("python" "scripts/calculate_pooled_eer.py" "$RESULTS_FOLDER" "$NORMALIZED_YAML" "$COMMENT")
    
    # Add each benchmark folder as a separate argument
    for subfolder in "${SUBDIRS[@]}"; do
        cmd_args+=("$subfolder")
    done
    
    # Debug: Show the command being executed (commented out to reduce output)
    # print_color "$WHITE" "  Executing: ${cmd_args[*]}"
    
    # Call the Python script for pooled EER calculation
    # Create temporary files to capture stdout and stderr separately
    local temp_stdout=$(mktemp)
    local temp_stderr=$(mktemp)
    
    # Execute command and capture streams
    "${cmd_args[@]}" > "$temp_stdout" 2> "$temp_stderr"
    local exit_code=$?
    
    # Read the captured streams
    local pooled_stdout=$(cat "$temp_stdout")
    local pooled_stderr=$(cat "$temp_stderr")
    
    # Clean up temporary files
    rm -f "$temp_stdout" "$temp_stderr"
    
    if [ $exit_code -eq 0 ] && [ ! -z "$pooled_stdout" ]; then
        # Use stdout for the numeric result
        local main_result="$pooled_stdout"
        
        # Check if we have a valid result format
        local result_parts=$(echo "$main_result" | wc -w)
        if [ "$result_parts" -eq 5 ]; then
            # Extract values from the result
            local pooled_min_score=$(echo "$main_result" | cut -d' ' -f1)
            local pooled_max_score=$(echo "$main_result" | cut -d' ' -f2)
            local pooled_threshold=$(echo "$main_result" | cut -d' ' -f3)
            local pooled_eer=$(echo "$main_result" | cut -d' ' -f4)
            local pooled_accuracy=$(echo "$main_result" | cut -d' ' -f5)
            
            # Add pooled EER to summary file
            echo "" >> "$SUMMARY_FILE"
            echo "POOLED_EER | $pooled_eer | $pooled_min_score | $pooled_max_score | $pooled_threshold | $pooled_accuracy" >> "$SUMMARY_FILE"
            
            # Display the detailed output from Python script (stderr)
            echo "$pooled_stderr" | while IFS= read -r line; do
                if [[ "$line" == *"✓"* ]]; then
                    print_color "$GREEN" "$line"
                elif [[ "$line" == *"  "* ]]; then
                    print_color "$WHITE" "$line"
                else
                    print_color "$CYAN" "$line"
                fi
            done
        else
            print_color "$RED" "❌ Invalid result format from pooled EER calculation"
            print_color "$YELLOW" "Output: $pooled_stdout"
        fi
    else
        print_color "$RED" "❌ Failed to calculate pooled EER"
        if [ ! -z "$pooled_stderr" ]; then
            print_color "$YELLOW" "Error details: $pooled_stderr"
        fi
    fi
}

# Function to calculate average EER
calculate_average_eer() {
    print_color "$CYAN" "🔄 Calculating average EER across datasets..."
    
    local total_eer=0
    local count=0
    local eer_values=()
    
    # Read individual EER values from summary file
    while IFS='|' read -r dataset eer rest; do
        dataset=$(echo "$dataset" | xargs)  # trim whitespace
        eer=$(echo "$eer" | xargs)  # trim whitespace
        
        # Skip header and empty lines, and exclude pooled EER if already calculated
        if [[ "$dataset" != "Dataset" && "$dataset" != "POOLED_EER" && "$dataset" != "AVERAGE_EER" && ! -z "$dataset" && ! -z "$eer" ]]; then
            # Validate EER is a number
            if [[ "$eer" =~ ^[0-9]*\.?[0-9]+$ ]]; then
                eer_values+=("$eer")
                total_eer=$(echo "$total_eer + $eer" | bc -l 2>/dev/null || echo "$total_eer + $eer" | awk '{print $1 + $3}')
                count=$((count + 1))
            fi
        fi
    done < "$SUMMARY_FILE"
    
    if [ $count -gt 0 ]; then
        local average_eer
        if command -v bc >/dev/null 2>&1; then
            average_eer=$(echo "scale=6; $total_eer / $count" | bc -l)
        else
            average_eer=$(awk "BEGIN {printf \"%.6f\", $total_eer / $count}")
        fi
        
        # Add average EER to summary file
        echo "AVERAGE_EER | $average_eer | - | - | - | -" >> "$SUMMARY_FILE"
        
        # Display average results
        print_color "$GREEN" "✓ Average EER Results (across $count datasets):"
        print_color "$WHITE" "  Average EER: $average_eer"
        print_color "$WHITE" "  Individual EERs: ${eer_values[*]}"
    else
        print_color "$RED" "❌ No valid EER values found for average calculation"
    fi
}

# Call the calculation functions
calculate_pooled_eer
calculate_average_eer

# Final summary
print_color "$MAGENTA" "┌─────────────────────────────────────────────────────────────────┐"
print_color "$MAGENTA" "│                       BENCHMARK COMPLETE                        │"
print_color "$MAGENTA" "└─────────────────────────────────────────────────────────────────┘"
print_color "$GREEN" "✓ All benchmarks completed successfully!"
print_color "$CYAN" "✓ Summary available at: $SUMMARY_FILE"

# Clean up any remaining temporary files
print_color "$CYAN" "🧹 Cleaning up temporary files..."
rm -f "$RESULTS_FOLDER"/temp_protocol_*.txt
rm -f "$RESULTS_FOLDER"/temp_scores_*.txt
rm -f "$RESULTS_FOLDER"/merged_scores_*.txt
rm -f /tmp/existing_scores_*.txt /tmp/protocol_eval_*.txt /tmp/protocol_ids_*.txt /tmp/existing_ids_*.txt /tmp/missing_ids_*.txt

# Pretty print the summary file
echo ""
print_color "$YELLOW" "📊 SUMMARY OF RESULTS:"
echo ""
print_color "$WHITE" "$(cat $SUMMARY_FILE | sed 's/|/│/g')"
echo ""
print_color "$GREEN" "Thanks for using the Bulk Benchmark Runner Tool!"