#!/bin/bash

# Bulk Benchmark Runner Script - Refactored Modular Version
# Main script that orchestrates benchmark execution using modular components

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source constants first (before other modules that depend on them)
source "$SCRIPT_DIR/benchmark_constants.sh"

# Source all modules
source "$SCRIPT_DIR/benchmark_utils.sh"
source "$SCRIPT_DIR/benchmark_config.sh"
source "$SCRIPT_DIR/benchmark_validation.sh"
source "$SCRIPT_DIR/benchmark_protocol.sh"
source "$SCRIPT_DIR/benchmark_scores.sh"
source "$SCRIPT_DIR/benchmark_execution.sh"
source "$SCRIPT_DIR/benchmark_eer.sh"
source "$SCRIPT_DIR/benchmark_merge.sh"

# Print banner
print_banner

# Parse command line arguments
parse_arguments "$@"

# Validate required arguments
validate_arguments

# Set default values
set_defaults

# Print configuration
print_color "$CYAN" "IS_RANDOM_START: $IS_RANDOM_START"

# Initialize results directory and summary file
initialize_results

# Get list of subdirectories
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
    print_color "$YELLOW" "Directory contents:"
    ls -la "$BENCHMARK_FOLDER"
    exit 1
fi

print_color "$CYAN" "✓ Starting benchmark with device $GPU_NUMBER and config $YAML_CONFIG"
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
    
    # Check if score file exists and is complete
    if validate_score_file "$SCORE_SAVE_PATH" "$PROTOCOL_PATH"; then
        # Run the scoring script
        if evaluate_results "$SCORE_SAVE_PATH" "$PROTOCOL_PATH" "$SUMMARY_FILE" "$subfolder_name"; then
            print_color "$GREEN" "✓ Results for $subfolder_name (using existing complete score file)"
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
                if evaluate_results "$SCORE_SAVE_PATH" "$PROTOCOL_PATH" "$SUMMARY_FILE" "$subfolder_name"; then
                    print_color "$GREEN" "✓ Results for $subfolder_name (using existing complete score file)"
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
    
    # Construct and execute benchmark command
    CMD=$(construct_benchmark_command \
        "$GPU_NUMBER" \
        "$YAML_CONFIG" \
        "$SCORE_PATH_TO_USE" \
        "$DATA_DIR" \
        "$PROTOCOL_TO_USE" \
        "$BASE_MODEL_PATH" \
        "$IS_BASE_MODEL_PATH_LN" \
        "$IS_RANDOM_START" \
        "$TRIM_LENGTH" \
        "${ADAPTER_PATHS:-}")
    
    execute_benchmark "$CMD"
    
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
    
    # Check if the final score file is complete and evaluate
    if validate_score_file "$SCORE_SAVE_PATH" "$PROTOCOL_PATH"; then
        evaluate_results "$SCORE_SAVE_PATH" "$PROTOCOL_PATH" "$SUMMARY_FILE" "$subfolder_name"
    elif [ -f "$SCORE_SAVE_PATH" ]; then
        print_color "$RED" "❌ Error: Score file exists but is incomplete/corrupted for $subfolder_name"
    else
        print_color "$RED" "❌ Error: Score file was not created for $subfolder_name"
    fi
    
    print_color "$GREEN" "✓ Finished processing $subfolder_name"
    echo ""
done

# Calculate pooled EER from all datasets
print_color "$MAGENTA" "┌─────────────────────────────────────────────────────────────────┐"
print_color "$MAGENTA" "│                    CALCULATING POOLED EER                       │"
print_color "$MAGENTA" "└─────────────────────────────────────────────────────────────────┘"

# Call the calculation functions
calculate_pooled_eer "$RESULTS_FOLDER" "$NORMALIZED_YAML" "$COMMENT" "$SUMMARY_FILE" "${SUBDIRS[@]}"
calculate_average_eer "$SUMMARY_FILE"

# Create merged protocol file for reuse
create_merged_protocol \
    "$RESULTS_FOLDER" \
    "$NORMALIZED_YAML" \
    "$COMMENT" \
    "$YAML_CONFIG" \
    "$BASE_MODEL_PATH" \
    "$SUMMARY_FILE" \
    "${SUBDIRS[@]}"

# Final summary
print_color "$MAGENTA" "┌─────────────────────────────────────────────────────────────────┐"
print_color "$MAGENTA" "│                       BENCHMARK COMPLETE                        │"
print_color "$MAGENTA" "└─────────────────────────────────────────────────────────────────┘"
print_color "$GREEN" "✓ All benchmarks completed successfully!"
print_color "$CYAN" "✓ Summary available at: $SUMMARY_FILE"
print_color "$CYAN" "✓ Merged protocol available at: $RESULTS_FOLDER/merged_protocol_${NORMALIZED_YAML}_${COMMENT}.txt"
print_color "$CYAN" "✓ Merged scores available at: $RESULTS_FOLDER/merged_scores_${NORMALIZED_YAML}_${COMMENT}.txt"
print_color "$CYAN" "✓ Protocol metadata available at: $RESULTS_FOLDER/pooled_merged_protocol_${NORMALIZED_YAML}_${COMMENT}.txt"

# Clean up any remaining temporary files
cleanup_temp_files "$RESULTS_FOLDER"

# Pretty print the summary file
echo ""
print_color "$YELLOW" "📊 SUMMARY OF RESULTS:"
echo ""
print_color "$WHITE" "$(cat $SUMMARY_FILE | sed 's/|/│/g')"
echo ""
print_color "$GREEN" "Thanks for using the Bulk Benchmark Runner Tool!"
