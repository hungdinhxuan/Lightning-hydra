#!/bin/bash

# Define colors for output
CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Placeholder print_color function
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Default paths
FUSION_DIR="/nvme1/hungdx/Lightning-hydra/logs/results/huggingface_benchrmark_Speech-DF-Arena/fused_scores"
PROTOCOL_DIR="/nvme1/hungdx/Lightning-hydra/data/huggingface_benchrmark_Speech-DF-Arena"
SUMMARY_FILE="/nvme1/hungdx/Lightning-hydra/logs/results/huggingface_benchrmark_Speech-DF-Arena/fusion_summary.txt"
SCORE_EVAL_SCRIPT="scripts/score_file_to_eer.py"

# Usage message
usage() {
    echo "Usage: $0 [-f fusion_dir] [-p protocol_dir] [-s summary_file] [-e eval_script]"
    echo "Options:"
    echo "  -f  Path to the fusion scores directory (default: $FUSION_DIR)"
    echo "  -p  Path to the protocol directory (default: $PROTOCOL_DIR)"
    echo "  -s  Path to the output summary file (default: $SUMMARY_FILE)"
    echo "  -e  Path to the score evaluation script (default: $SCORE_EVAL_SCRIPT)"
    echo "  -h  Display this help message"
    echo "Example:"
    echo "  $0 -f /path/to/fused_scores -p /path/to/protocols -s /path/to/summary.txt -e /path/to/score_file_to_eer.py"
    exit 1
}

# Parse command-line arguments
while getopts "f:p:s:e:h" opt; do
    case $opt in
        f) FUSION_DIR="$OPTARG" ;;
        p) PROTOCOL_DIR="$OPTARG" ;;
        s) SUMMARY_FILE="$OPTARG" ;;
        e) SCORE_EVAL_SCRIPT="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Ensure the summary file directory exists
mkdir -p "$(dirname "$SUMMARY_FILE")"

# Clear or create the summary file with headers
echo "Dataset | EER | Min Score | Max Score | Threshold | Accuracy" > "$SUMMARY_FILE"

# Check if fusion directory exists
if [ ! -d "$FUSION_DIR" ]; then
    print_color "$RED" "❌ Error: Fusion directory $FUSION_DIR does not exist"
    exit 1
fi

# Check if protocol directory exists
if [ ! -d "$PROTOCOL_DIR" ]; then
    print_color "$RED" "❌ Error: Protocol directory $PROTOCOL_DIR does not exist"
    exit 1
fi

# Check if score evaluation script exists
if [ ! -f "$SCORE_EVAL_SCRIPT" ]; then
    print_color "$RED" "❌ Error: Score evaluation script $SCORE_EVAL_SCRIPT does not exist"
    exit 1
fi

# Iterate through score files in the fusion directory
found_files=false
for score_file in "$FUSION_DIR"/*_fused.txt; do
    if [ -f "$score_file" ]; then
        found_files=true
        # Extract dataset name from score file (remove _fused.txt suffix)
        dataset_name=$(basename "$score_file" | sed 's/_fused\.txt$//')

        # Assume protocol file is named <dataset_name>/protocol.txt in PROTOCOL_DIR
        protocol_file="$PROTOCOL_DIR/$dataset_name/protocol.txt"

        print_color "$CYAN" "🔄 Evaluating $dataset_name..."

        # Check if protocol file exists
        if [ ! -f "$protocol_file" ]; then
            print_color "$RED" "❌ Error: Protocol file $protocol_file not found for $dataset_name"
            continue
        fi

        # Run the scoring script
        RESULT=$(python "$SCORE_EVAL_SCRIPT" "$score_file" "$protocol_file")

        # Check if the evaluation script was successful
        if [ $? -eq 0 ]; then
            # Extract values from the result
            MIN_SCORE=$(echo "$RESULT" | cut -d' ' -f1)
            MAX_SCORE=$(echo "$RESULT" | cut -d' ' -f2)
            THRESHOLD=$(echo "$RESULT" | cut -d' ' -f3)
            EER=$(echo "$RESULT" | cut -d' ' -f4)
            ACCURACY=$(echo "$RESULT" | cut -d' ' -f5)

            # Format output for summary file
            echo "$dataset_name | $EER | $MIN_SCORE | $MAX_SCORE | $THRESHOLD | $ACCURACY" >> "$SUMMARY_FILE"

            # Display results
            print_color "$GREEN" "✓ Results for $dataset_name:"
            print_color "$WHITE" "  EER      : $EER"
            print_color "$WHITE" "  Accuracy : $ACCURACY"
            print_color "$WHITE" "  Threshold: $THRESHOLD"
            print_color "$WHITE" "  Min Score: $MIN_SCORE"
            print_color "$WHITE" "  Max Score: $MAX_SCORE"
        else
            print_color "$RED" "❌ Error: Failed to evaluate results for $dataset_name"
        fi
    fi
done

# Check if any score files were found
if [ "$found_files" = false ]; then
    print_color "$RED" "❌ No score files found in $FUSION_DIR"
    exit 1
fi

print_color "$GREEN" "✅ Evaluation completed! Summary saved to $SUMMARY_FILE"