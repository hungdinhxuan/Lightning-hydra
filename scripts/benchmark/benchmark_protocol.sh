#!/bin/bash

# Benchmark Protocol Module
# Handles protocol file operations including missing entry detection

# Source utilities and constants
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/benchmark_utils.sh"
source "$SCRIPT_DIR/benchmark_constants.sh"

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
    
    # Create temporary file for protocol subset
    local temp_id="$$_$(date +%s)_$RANDOM"
    local temp_protocol_subset="/tmp/${TEMP_PROTOCOL_EVAL_PREFIX}_${temp_id}.txt"
    
    # Extract subset entries from protocol file
    if should_use_protocol_subset "$protocol_file"; then
        # If protocol has the specified subset, use only subset lines
        grep "$PROTOCOL_SUBSET" "$protocol_file" > "$temp_protocol_subset"
        print_color "$WHITE" "  Using protocol subset: $PROTOCOL_SUBSET"
    else
        # If no subset found or subset is empty, use all lines
        cp "$protocol_file" "$temp_protocol_subset"
        print_color "$WHITE" "  Using all protocol lines (subset '$(get_protocol_subset_name)' not found or not specified)"
    fi
    
    local total_subset_lines=$(wc -l < "$temp_protocol_subset")
    print_color "$WHITE" "  Total $(get_protocol_subset_name) lines in protocol: $total_subset_lines"
    
    # Calculate missing lines (sequential evaluation - just skip processed lines)
    local missing_count=$((total_subset_lines - existing_lines))
    
    if [ $missing_count -le 0 ]; then
        print_color "$GREEN" "  No missing entries found."
        rm -f "$temp_protocol_subset"
        touch "$temp_protocol"  # Create empty temp protocol
        return 0
    fi
    
    # Create temporary protocol file with remaining entries (starting from existing_lines + 1)
    local start_line=$((existing_lines + 1))
    tail -n +$start_line "$temp_protocol_subset" > "$temp_protocol"
    
    print_color "$YELLOW" "  Found $missing_count missing entries (starting from line $start_line)"
    
    # Clean up temporary files
    rm -f "$temp_protocol_subset"
    
    return $missing_count
}

# Function to extract protocol subset from protocol
extract_eval_subset() {
    local protocol_file="$1"
    local output_file="$2"
    
    if should_use_protocol_subset "$protocol_file"; then
        # If protocol has the specified subset, use only subset lines
        grep "$PROTOCOL_SUBSET" "$protocol_file" > "$output_file"
    else
        # If no subset found or subset is empty, use all lines
        cp "$protocol_file" "$output_file"
    fi
}
