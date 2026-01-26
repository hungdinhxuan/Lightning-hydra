#!/bin/bash

# MDT Paper Benchmark Runner Script (based on benchmark.sh)
# Runs 4 durations (1s, 2s, 3s, 4s) and variable-length eval for each dataset

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
progress_bar() {
    local current=$1
    local total=$2
    local width=50
    local percent=$((current * 100 / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))
    printf "\r${BLUE}Progress: ["
    printf "%${filled}s" | tr ' ' '#'
    printf "%${empty}s" | tr ' ' '-'
    printf "] %d%%${NC}" $percent
}

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

# Function to run benchmark and handle scoring
run_benchmark_with_validation() {
    local setting="$1"
    local score_save_path="$2"
    local protocol_path="$3"
    local cmd="$4"
    local subfolder_name="$5"
    
    # Initialize variables for protocol and score path handling
    local protocol_to_use="$protocol_path"
    local score_path_to_use="$score_save_path"
    local use_temp_protocol=false
    local temp_protocol_path=""
    local temp_score_path=""
    local random_id=""
    
    # Check if score file exists and is complete
    if validate_score_file "$score_save_path" "$protocol_path"; then
        # Run the scoring script
        print_color "$CYAN" "🔄 Evaluating existing complete results for $setting..."
        RESULT=$(python scripts/score_file_to_eer.py "$score_save_path" "$protocol_path")
        
        # Check if the evaluation script was successful
        if [ $? -eq 0 ]; then
            # Extract values from the result
            MIN_SCORE=$(echo "$RESULT" | cut -d' ' -f1)
            MAX_SCORE=$(echo "$RESULT" | cut -d' ' -f2)
            THRESHOLD=$(echo "$RESULT" | cut -d' ' -f3)
            EER=$(echo "$RESULT" | cut -d' ' -f4)
            ACCURACY=$(echo "$RESULT" | cut -d' ' -f5)
            
            # Format output for summary file
            echo "$subfolder_name | $setting | $EER | $MIN_SCORE | $MAX_SCORE | $THRESHOLD | $ACCURACY" >> "$SUMMARY_FILE"
            
            # Display results
            print_color "$GREEN" "✓ Results for $subfolder_name ($setting) - using existing complete score file:"
            print_color "$WHITE" "  EER      : $EER"
            print_color "$WHITE" "  Accuracy : $ACCURACY"
            print_color "$WHITE" "  Threshold: $THRESHOLD"
            print_color "$WHITE" "  Min Score: $MIN_SCORE"
            print_color "$WHITE" "  Max Score: $MAX_SCORE"
            return
        else
            print_color "$RED" "❌ Error: Failed to evaluate results for $subfolder_name ($setting)"
        fi
    elif [ -f "$score_save_path" ]; then
        print_color "$YELLOW" "⚠️ Warning: Score file exists but is incomplete/corrupted for $subfolder_name ($setting)."
        
        # Create temporary protocol file with missing entries (using random ID to avoid conflicts)
        random_id="$$_$(date +%s)_$RANDOM"
        temp_protocol_path="$RESULTS_FOLDER/temp_protocol_${subfolder_name}_${setting}_${random_id}.txt"
        
        create_missing_protocol "$score_save_path" "$protocol_path" "$temp_protocol_path"
        missing_count=$?
        
        if [ $missing_count -gt 0 ]; then
            print_color "$CYAN" "🔄 Running benchmark for $missing_count missing entries only..."
            protocol_to_use="$temp_protocol_path"
            temp_score_path="$RESULTS_FOLDER/temp_scores_${subfolder_name}_${setting}_${random_id}.txt"
            score_path_to_use="$temp_score_path"
            use_temp_protocol=true
        elif [ $missing_count -eq 0 ]; then
            print_color "$GREEN" "✓ No missing entries found. Score file is actually complete."
            # Re-validate to make sure
            if validate_score_file "$score_save_path" "$protocol_path"; then
                # Run the scoring script directly since file is complete
                print_color "$CYAN" "🔄 Evaluating existing complete results..."
                RESULT=$(python scripts/score_file_to_eer.py "$score_save_path" "$protocol_path")
                
                if [ $? -eq 0 ]; then
                    # Extract values from the result
                    MIN_SCORE=$(echo "$RESULT" | cut -d' ' -f1)
                    MAX_SCORE=$(echo "$RESULT" | cut -d' ' -f2)
                    THRESHOLD=$(echo "$RESULT" | cut -d' ' -f3)
                    EER=$(echo "$RESULT" | cut -d' ' -f4)
                    ACCURACY=$(echo "$RESULT" | cut -d' ' -f5)
                    
                    # Format output for summary file
                    echo "$subfolder_name | $setting | $EER | $MIN_SCORE | $MAX_SCORE | $THRESHOLD | $ACCURACY" >> "$SUMMARY_FILE"
                    
                    # Display results
                    print_color "$GREEN" "✓ Results for $subfolder_name ($setting) - using existing complete score file:"
                    print_color "$WHITE" "  EER      : $EER"
                    print_color "$WHITE" "  Accuracy : $ACCURACY"
                    print_color "$WHITE" "  Threshold: $THRESHOLD"
                    print_color "$WHITE" "  Min Score: $MIN_SCORE"
                    print_color "$WHITE" "  Max Score: $MAX_SCORE"
                fi
            fi
            return
        else
            print_color "$RED" "❌ Error: Failed to analyze missing entries. Re-running full benchmark..."
            protocol_to_use="$protocol_path"
            score_path_to_use="$score_save_path"
            use_temp_protocol=false
        fi
    else
        print_color "$CYAN" "ℹ️ No existing score file found for $subfolder_name ($setting). Running fresh benchmark..."
        protocol_to_use="$protocol_path"
        score_path_to_use="$score_save_path"
        use_temp_protocol=false
    fi
    
    # Check if protocol file exists
    if [ ! -f "$protocol_to_use" ]; then
        print_color "$RED" "⚠️ Warning: Protocol file not found at $protocol_to_use. Skipping this setting."
        return
    fi
    
    # Update the command with the correct paths
    local updated_cmd=$(echo "$cmd" | sed "s|++model.score_save_path=\"[^\"]*\"|++model.score_save_path=\"$score_path_to_use\"|")
    updated_cmd=$(echo "$updated_cmd" | sed "s|++data.args.protocol_path=\"[^\"]*\"|++data.args.protocol_path=\"$protocol_to_use\"|")
    
    # Execute the command
    print_color "$CYAN" "🔄 Running benchmark for $setting..."
    print_color "$WHITE" "$updated_cmd"
    echo ""
    
    # Execute the command with a spinner
    eval $updated_cmd &
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
    if [ "$use_temp_protocol" = true ]; then
        if [ -f "$temp_score_path" ]; then
            print_color "$CYAN" "🔄 Merging temporary scores with existing scores..."
            merged_score_path="$RESULTS_FOLDER/merged_scores_${subfolder_name}_${setting}_${random_id}.txt"
            merge_score_files "$score_save_path" "$temp_score_path" "$merged_score_path"
            
            # Clean up temporary files
            rm -f "$temp_protocol_path" "$temp_score_path" "$merged_score_path"
            print_color "$GREEN" "✓ Temporary files cleaned up"
        else
            print_color "$RED" "❌ Error: Temporary score file was not created for $subfolder_name ($setting)"
        fi
    fi
    
    # Check if the final score file is complete
    if validate_score_file "$score_save_path" "$protocol_path"; then
        # Run the scoring script
        print_color "$CYAN" "🔄 Evaluating results..."
        RESULT=$(python scripts/score_file_to_eer.py "$score_save_path" "$protocol_path")
        
        # Check if the evaluation script was successful
        if [ $? -eq 0 ]; then
            # Extract values from the result
            MIN_SCORE=$(echo "$RESULT" | cut -d' ' -f1)
            MAX_SCORE=$(echo "$RESULT" | cut -d' ' -f2)
            THRESHOLD=$(echo "$RESULT" | cut -d' ' -f3)
            EER=$(echo "$RESULT" | cut -d' ' -f4)
            ACCURACY=$(echo "$RESULT" | cut -d' ' -f5)
            
            # Format output for summary file
            echo "$subfolder_name | $setting | $EER | $MIN_SCORE | $MAX_SCORE | $THRESHOLD | $ACCURACY" >> "$SUMMARY_FILE"
            
            # Display results
            print_color "$GREEN" "✓ Results for $subfolder_name ($setting):"
            print_color "$WHITE" "  EER      : $EER"
            print_color "$WHITE" "  Accuracy : $ACCURACY"
            print_color "$WHITE" "  Threshold: $THRESHOLD"
            print_color "$WHITE" "  Min Score: $MIN_SCORE"
            print_color "$WHITE" "  Max Score: $MAX_SCORE"
        else
            print_color "$RED" "❌ Error: Failed to evaluate results for $subfolder_name ($setting)"
        fi
    elif [ -f "$score_save_path" ]; then
        print_color "$RED" "❌ Error: Score file exists but is incomplete/corrupted for $subfolder_name ($setting)"
    else
        print_color "$RED" "❌ Error: Score file was not created for $subfolder_name ($setting)"
    fi
    
    print_color "$GREEN" "✓ Finished processing $subfolder_name ($setting)"
    echo ""
}

# Usage info
show_usage() {
    print_color "$BLUE" "\nMDT Paper Benchmark Runner Script"
    print_color "$CYAN" "Usage: $0 -g <gpu_number> -c <yaml_config_file> -b <bulk_benchmark_folder> -m <base_model_path> -r <results_folder> -n <comment> [-a <adapter_paths>] [-l <is_base_model_path_ln>] [--random_start True|False]"
    print_color "$YELLOW" "\nParameters:"
    echo "  -g <gpu_number>             GPU number to use (0, 1, 2, 3, ...)"
    echo "  -c <yaml_config_file>       Yaml config file path (e.g., cnsl/xlsr_vib_large_corpus)"
    echo "  -b <bulk_benchmark_folder>  Bulk benchmark folder path"
    echo "  -m <base_model_path>        Base model path"
    echo "  -r <results_folder>         Results folder path"
    echo "  -n <comment>                Comment to note"
    echo "  -a <adapter_paths>          Adapter paths (optional)"
    echo "  -l <is_base_model_path_ln>  Whether to use Lightning checkpoint loading (default: true)"
    echo "  --random_start <True|False> Set random_start (default: False)"
    exit 1
}

# Parse arguments
RANDOM_START="False"
while [[ $# -gt 0 ]]; do
    case $1 in
        -g) GPU_NUMBER="$2"; shift 2;;
        -c) YAML_CONFIG="$2"; shift 2;;
        -b) BENCHMARK_FOLDER="$2"; shift 2;;
        -m) BASE_MODEL_PATH="$2"; shift 2;;
        -r) RESULTS_FOLDER="$2"; shift 2;;
        -n) COMMENT="$2"; shift 2;;
        -a) ADAPTER_PATHS="$2"; shift 2;;
        -l) IS_BASE_MODEL_PATH_LN="$2"; shift 2;;
        --random_start) RANDOM_START="$2"; shift 2;;
        *) show_usage;;
    esac
done

# Check required args
if [ -z "$GPU_NUMBER" ] || [ -z "$YAML_CONFIG" ] || [ -z "$BENCHMARK_FOLDER" ] || [ -z "$BASE_MODEL_PATH" ] || [ -z "$RESULTS_FOLDER" ]; then
    print_color "$RED" "Error: Missing required parameters"
    show_usage
fi

if [ -z "$IS_BASE_MODEL_PATH_LN" ]; then
    IS_BASE_MODEL_PATH_LN="true"
fi

# Prepare results dir
RESULTS_FOLDER="${RESULTS_FOLDER%/}/${COMMENT}"
mkdir -p "$RESULTS_FOLDER"

# Summary file
SUMMARY_FILE="$RESULTS_FOLDER/summary_results.txt"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
NORMALIZED_YAML=$(echo "$YAML_CONFIG" | tr '/' '_')

echo "Config: $YAML_CONFIG" > "$SUMMARY_FILE"
echo "Base_model_path: $BASE_MODEL_PATH" >> "$SUMMARY_FILE"
echo "Lora Path: ${ADAPTER_PATHS:-None}" >> "$SUMMARY_FILE"
echo "Is Base Model Path LN: $IS_BASE_MODEL_PATH_LN" >> "$SUMMARY_FILE"
echo "Random Start: $RANDOM_START" >> "$SUMMARY_FILE"
echo "Date: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Dataset | Setting | EER | min_score | max_score | Threshold | Accuracy" >> "$SUMMARY_FILE"

# Durations and settings
DURATIONS=(16000 32000 48000 64600)
DURATION_LABELS=(1s 2s 3s 4s)

# List subdirs
declare -a SUBDIRS=()
while IFS= read -r dir; do
    if [[ -d "$BENCHMARK_FOLDER/$dir" ]]; then
        SUBDIRS+=("$BENCHMARK_FOLDER/$dir")
    fi
done < <(ls -1 "$BENCHMARK_FOLDER")

TOTAL_SUBFOLDERS=${#SUBDIRS[@]}
if [ $TOTAL_SUBFOLDERS -eq 0 ]; then
    print_color "$RED" "Error: No subdirectories found in '$BENCHMARK_FOLDER'."
    exit 1
fi

CURRENT_SUBFOLDER=0
for subfolder in "${SUBDIRS[@]}"; do
    CURRENT_SUBFOLDER=$((CURRENT_SUBFOLDER + 1))
    subfolder_name=$(basename "$subfolder")
    print_color "$YELLOW" "\nProcessing dataset: $subfolder_name ($CURRENT_SUBFOLDER/$TOTAL_SUBFOLDERS)"
    progress_bar $CURRENT_SUBFOLDER $TOTAL_SUBFOLDERS
    DATA_DIR="$subfolder"
    PROTOCOL_PATH="$subfolder/protocol.txt"

    # 1. Fixed durations, no_pad=False
    for idx in ${!DURATIONS[@]}; do
        CUT_VAL=${DURATIONS[$idx]}
        LABEL=${DURATION_LABELS[$idx]}
        SETTING="${LABEL}_no_pad"
        SCORE_SAVE_PATH="$RESULTS_FOLDER/${subfolder_name}_${NORMALIZED_YAML}_${COMMENT}_${SETTING}.txt"
        CMD="CUDA_VISIBLE_DEVICES=$GPU_NUMBER python src/train.py experiment=$YAML_CONFIG ++model.score_save_path=\"$SCORE_SAVE_PATH\" ++data.data_dir=\"$DATA_DIR\" ++data.args.protocol_path=\"$PROTOCOL_PATH\" ++train=False ++test=True ++model.spec_eval=True ++data.batch_size=128 ++model.base_model_path=\"$BASE_MODEL_PATH\" ++model.is_base_model_path_ln=$IS_BASE_MODEL_PATH_LN ++data.args.trim_length=$CUT_VAL ++data.args.random_start=$RANDOM_START ++data.args.no_pad=False"
        if [ ! -z "$ADAPTER_PATHS" ]; then
            CMD+=" ++model.adapter_paths=\"$ADAPTER_PATHS\""
        fi
        
        run_benchmark_with_validation "$SETTING" "$SCORE_SAVE_PATH" "$PROTOCOL_PATH" "$CMD" "$subfolder_name"
    done

    # 2. Variable length eval: no_pad=True, cut=64600
    SETTING="varlen_no_pad"
    SCORE_SAVE_PATH="$RESULTS_FOLDER/${subfolder_name}_${NORMALIZED_YAML}_${COMMENT}_${SETTING}.txt"
    CMD="CUDA_VISIBLE_DEVICES=$GPU_NUMBER python src/train.py experiment=$YAML_CONFIG ++model.score_save_path=\"$SCORE_SAVE_PATH\" ++data.data_dir=\"$DATA_DIR\" ++data.args.protocol_path=\"$PROTOCOL_PATH\" ++train=False ++test=True ++model.spec_eval=True ++data.batch_size=1 ++model.base_model_path=\"$BASE_MODEL_PATH\" ++model.is_base_model_path_ln=$IS_BASE_MODEL_PATH_LN ++data.args.cut=64600 ++data.args.random_start=$RANDOM_START ++data.args.no_pad=True"
    if [ ! -z "$ADAPTER_PATHS" ]; then
        CMD+=" ++model.adapter_paths=\"$ADAPTER_PATHS\""
    fi
    
    run_benchmark_with_validation "$SETTING" "$SCORE_SAVE_PATH" "$PROTOCOL_PATH" "$CMD" "$subfolder_name"

done

# Clean up any remaining temporary files
print_color "$CYAN" "🧹 Cleaning up temporary files..."
rm -f "$RESULTS_FOLDER"/temp_protocol_*.txt
rm -f "$RESULTS_FOLDER"/temp_scores_*.txt
rm -f "$RESULTS_FOLDER"/merged_scores_*.txt
rm -f /tmp/existing_scores_*.txt /tmp/protocol_eval_*.txt /tmp/protocol_ids_*.txt /tmp/existing_ids_*.txt /tmp/missing_ids_*.txt

print_color "$GREEN" "\n✓ All benchmarks completed!"
print_color "$CYAN" "✓ Summary available at: $SUMMARY_FILE"
print_color "$YELLOW" "\n📊 SUMMARY OF RESULTS:"
print_color "$WHITE" "$(cat $SUMMARY_FILE | sed 's/|/│/g')"
