#!/bin/bash

# Benchmark Utilities Module
# Provides color output, progress tracking, and UI functions

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
    
    local width=$PROGRESS_BAR_WIDTH
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
    local script_name="${0##*/}"
    print_color "$BLUE" "┌─────────────────────────────────────────────────────────────────┐"
    print_color "$BLUE" "│                 Bulk Benchmark Runner Script                    │"
    print_color "$BLUE" "└─────────────────────────────────────────────────────────────────┘"
    echo ""
    print_color "$CYAN" "Usage: $script_name -g <gpu_identifier> -c <yaml_config_file> -b <bulk_benchmark_folder> -m <base_model_path> -r <results_folder> -n <comment> [-a <adapter_paths>] [-l <is_base_model_path_ln>] [-s <is_random_start>] [-t <trim_length>]"
    echo ""
    print_color "$YELLOW" "Parameters:"
    echo "  -g <gpu_identifier>         GPU identifier (index like 0/1 or MIG UUID like MIG-xxxx)"
    echo "  -c <yaml_config_file>       Yaml config file path (e.g., cnsl/xlsr_vib_large_corpus)"
    echo "  -b <bulk_benchmark_folder>  Bulk benchmark folder path"
    echo "  -m <base_model_path>        Base model path"
    echo "  -r <results_folder>         Results folder path"
    echo "  -n <comment>                Comment to note"
    echo "  -a <adapter_paths>          Adapter paths (optional)"
    echo "  -l <is_base_model_path_ln>  Whether to use Lightning checkpoint loading (default: true)"
    echo "  -s <is_random_start>        Whether to use random start (default: true)"
    echo "  -t <trim_length>            Trim length for data processing (default: 64000)"
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

# Function to display a spinner while command is running
run_with_spinner() {
    local cmd="$1"
    local description="${2:-Processing}"
    
    # Execute the command in background
    eval "$cmd" &
    local PID=$!
    
    # Display a spinner while the command is running
    local spin='-\|/'
    local i=0
    while kill -0 $PID 2>/dev/null; do
        i=$(( (i+1) % 4 ))
        printf "\r${CYAN}⏳ %s: %c${RESET}" "$description" "${spin:$i:1}"
        sleep .1
    done
    wait $PID
    local exit_code=$?
    printf "\r${GREEN}✓ %s completed                 ${RESET}\n" "$description"
    return $exit_code
}

# Function to clean up temporary files
cleanup_temp_files() {
    local results_folder="${1:-}"
    print_color "$CYAN" "🧹 Cleaning up temporary files..."
    
    if [ -n "$results_folder" ]; then
        rm -f "$results_folder"/temp_protocol_*.txt
        rm -f "$results_folder"/temp_scores_*.txt
    fi
    
    rm -f /tmp/temp_protocol_*.txt /tmp/temp_scores_*.txt
    rm -f /tmp/existing_scores_*.txt /tmp/protocol_eval_*.txt
    rm -f /tmp/protocol_ids_*.txt /tmp/existing_ids_*.txt /tmp/missing_ids_*.txt
}
