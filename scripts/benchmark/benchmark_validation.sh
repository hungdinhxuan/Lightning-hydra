#!/bin/bash

# Benchmark Validation Module
# Handles score file validation

# Source utilities and constants
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/benchmark_utils.sh"
source "$SCRIPT_DIR/benchmark_constants.sh"

# Function to validate score file completeness
validate_score_file() {
    local score_file="$1"
    local protocol_file="$2"
    
    if [ ! -f "$score_file" ]; then
        return 1  # Score file doesn't exist
    fi
    
    # Count lines in score file (excluding empty lines)
    local score_lines=$(grep -c "^[^[:space:]]*[[:space:]]" "$score_file" 2>/dev/null || echo "0")
    
    # Count subset lines in protocol file
    local subset_lines=0
    if should_use_protocol_subset "$protocol_file"; then
        subset_lines=$(grep -c "$PROTOCOL_SUBSET" "$protocol_file" 2>/dev/null || echo "0")
    fi
    
    # If no subset found or subset is empty, count all lines (fallback)
    if [ "$subset_lines" -eq 0 ]; then
        subset_lines=$(grep -c "^[^[:space:]]*[[:space:]]" "$protocol_file" 2>/dev/null || echo "0")
    fi
    
    print_color "$WHITE" "  Score file lines: $score_lines"
    print_color "$WHITE" "  Expected lines ($(get_protocol_subset_name) subset): $subset_lines"
    
    if [ "$score_lines" -eq "$subset_lines" ] && [ "$score_lines" -gt 0 ]; then
        return 0  # Valid and complete
    else
        return 2  # Incomplete or corrupted
    fi
}
