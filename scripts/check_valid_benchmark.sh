#!/bin/bash

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Function to validate protocol.txt format
validate_protocol() {
    local file=$1
    local folder=$2
    local errors=0

    # Check if file exists
    if [[ ! -f "$file" ]]; then
        echo -e "${RED}ERROR: $folder/protocol.txt does not exist${NC}"
        return 1
    fi

    # Check if file is empty
    if [[ ! -s "$file" ]]; then
        echo -e "${RED}ERROR: $folder/protocol.txt is empty${NC}"
        return 1
    fi

    # Validate each line format: <relative_path> <subset> <label>
    # Expected: <path> eval (bonafide|spoof)
    while IFS=' ' read -r path subset label; do
        # Skip empty lines
        [[ -z "$path" && -z "$subset" && -z "$label" ]] && continue

        # Check if line has exactly 3 fields
        if [[ -z "$path" || -z "$subset" || -z "$label" ]]; then
            echo -e "${RED}ERROR: Invalid line format in $folder/protocol.txt: '$path $subset $label'${NC}"
            ((errors++))
            continue
        fi

        # Validate label (should be bonafide or spoof)
        if [[ "$label" != "bonafide" && "$label" != "spoof" ]]; then
            echo -e "${RED}ERROR: Invalid label in $folder/protocol.txt: '$label' (expected 'bonafide' or 'spoof')${NC}"
            ((errors++))
        fi

        # Validate subset (assuming 'eval' is expected, adjust if needed)
        if [[ "$subset" != "eval" ]]; then
            echo -e "${YELLOW}WARNING: Unexpected subset in $folder/protocol.txt: '$subset' (expected 'eval')${NC}"
        fi

        # # Validate relative_path format (basic check for <folder>/<file>.wav)
        # if ! [[ "$path" =~ ^[^/]+/[^/]+\.wav$ ]]; then
        #     echo -e "${RED}ERROR: Invalid path format in $folder/protocol.txt: '$path' (expected '<folder>/<file>.wav')${NC}"
        #     ((errors++))
        # fi
    done < "$file"

    if [[ $errors -eq 0 ]]; then
        echo -e "${GREEN}SUCCESS: $folder/protocol.txt is valid${NC}"
        return 0
    else
        echo -e "${RED}ERROR: $folder/protocol.txt has $errors invalid lines${NC}"
        return 1
    fi
}

# Check if BENCHMARK_ROOT is provided
if [[ -z "$1" ]]; then
    echo -e "${RED}ERROR: Please provide BENCHMARK_ROOT as a parameter${NC}"
    echo -e "Usage: $0 /path/to/benchmark_root"
    exit 1
fi

BENCHMARK_ROOT="$1"

# Check if BENCHMARK_ROOT exists
if [[ ! -d "$BENCHMARK_ROOT" ]]; then
    echo -e "${RED}ERROR: Directory $BENCHMARK_ROOT does not exist${NC}"
    exit 1
fi

# Get list of directories in BENCHMARK_ROOT
mapfile -t folders < <(ls -d "$BENCHMARK_ROOT"/*/ 2>/dev/null | sed "s|$BENCHMARK_ROOT/||g" | sed 's|/$||')

if [[ ${#folders[@]} -eq 0 ]]; then
    echo -e "${RED}ERROR: No folders found in $BENCHMARK_ROOT${NC}"
    exit 1
fi

echo -e "${BLUE}Checking protocol.txt files in $BENCHMARK_ROOT${NC}"
echo -e "${BLUE}Found ${#folders[@]} folders: ${folders[*]}${NC}"

# Track valid and invalid folders
valid_count=0
invalid_count=0

# Process each folder with a progress bar
for i in "${!folders[@]}"; do
    folder="${folders[i]}"
    progress_bar $((i + 1)) ${#folders[@]}
    validate_protocol "$BENCHMARK_ROOT/$folder/protocol.txt" "$folder"
    if [[ $? -eq 0 ]]; then
        ((valid_count++))
    else
        ((invalid_count++))
    fi
done
echo # Newline after progress bar

# Summary
echo -e "${BLUE}Summary:${NC}"
echo -e "${GREEN}Valid protocol.txt files: $valid_count${NC}"
echo -e "${RED}Invalid or missing protocol.txt files: $invalid_count${NC}"

if [[ $invalid_count -eq 0 ]]; then
    echo -e "${GREEN}All protocol.txt files are valid!${NC}"
else
    echo -e "${YELLOW}Some protocol.txt files are missing or invalid. Please check the errors above.${NC}"
fi