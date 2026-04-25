#!/usr/bin/env python3
"""
Debug script to check validation logic

Usage:
    PROTOCOL_SUBSET=dev python scripts/benchmark_py/debug_validation.py \
        /path/to/score_file.txt \
        /path/to/protocol.txt
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_py.constants import CONSTANTS
from benchmark_py.validation import validate_score_file
from benchmark_py.protocol import create_missing_protocol
from benchmark_py.utils import print_color, Color


def debug_validation(score_file: Path, protocol_file: Path):
    """Debug validation and protocol logic"""
    
    print_color(Color.MAGENTA, "=" * 70)
    print_color(Color.MAGENTA, "VALIDATION DEBUG TOOL")
    print_color(Color.MAGENTA, "=" * 70)
    print()
    
    # Check files exist
    print_color(Color.CYAN, "1. Checking files...")
    print_color(Color.WHITE, f"  Score file: {score_file}")
    print_color(Color.WHITE, f"    Exists: {score_file.exists()}")
    print_color(Color.WHITE, f"  Protocol file: {protocol_file}")
    print_color(Color.WHITE, f"    Exists: {protocol_file.exists()}")
    print()
    
    if not protocol_file.exists():
        print_color(Color.RED, "Protocol file not found!")
        return
    
    # Show configuration
    print_color(Color.CYAN, "2. Configuration...")
    print_color(Color.WHITE, f"  PROTOCOL_SUBSET: '{CONSTANTS.protocol_subset}'")
    print_color(Color.WHITE, f"  Should use subset: {CONSTANTS.should_use_protocol_subset(str(protocol_file))}")
    print()
    
    # Count score lines
    print_color(Color.CYAN, "3. Score file analysis...")
    if score_file.exists():
        score_lines = 0
        with open(score_file, 'r') as f:
            for i, line in enumerate(f, 1):
                line_stripped = line.strip()
                if line_stripped and not line_stripped.startswith('#'):
                    score_lines += 1
                    if i <= 3:
                        print_color(Color.WHITE, f"  Line {i}: {line_stripped[:80]}...")
        print_color(Color.WHITE, f"  Total score lines: {score_lines}")
    else:
        print_color(Color.YELLOW, "  Score file does not exist")
    print()
    
    # Count protocol lines
    print_color(Color.CYAN, "4. Protocol file analysis...")
    total_lines = 0
    subset_lines = 0
    with open(protocol_file, 'r') as f:
        for i, line in enumerate(f, 1):
            line_stripped = line.strip()
            if line_stripped and not line_stripped.startswith('#'):
                total_lines += 1
                if CONSTANTS.protocol_subset and CONSTANTS.protocol_subset in line:
                    subset_lines += 1
                if i <= 3:
                    has_subset = CONSTANTS.protocol_subset in line if CONSTANTS.protocol_subset else False
                    marker = " [HAS SUBSET]" if has_subset else ""
                    print_color(Color.WHITE, f"  Line {i}: {line_stripped[:80]}...{marker}")
    
    print_color(Color.WHITE, f"  Total lines: {total_lines}")
    print_color(Color.WHITE, f"  Lines with subset '{CONSTANTS.protocol_subset}': {subset_lines}")
    print()
    
    # Validate
    print_color(Color.CYAN, "5. Validation result...")
    result = validate_score_file(score_file, protocol_file, verbose=True)
    print_color(Color.WHITE, f"  Is valid: {result.is_valid}")
    print_color(Color.WHITE, f"  Message: {result.message}")
    print()
    
    # Test missing protocol creation
    if score_file.exists() and not result.is_valid:
        print_color(Color.CYAN, "6. Testing missing protocol creation...")
        temp_protocol = Path(f"/tmp/debug_temp_protocol_{score_file.stem}.txt")
        missing_count = create_missing_protocol(score_file, protocol_file, temp_protocol)
        
        print_color(Color.WHITE, f"  Missing count: {missing_count}")
        if missing_count > 0 and temp_protocol.exists():
            print_color(Color.WHITE, f"  Temp protocol created at: {temp_protocol}")
            with open(temp_protocol, 'r') as f:
                lines = f.readlines()
                print_color(Color.WHITE, f"  Temp protocol has {len(lines)} lines")
                if lines:
                    print_color(Color.WHITE, "  First 3 lines:")
                    for i, line in enumerate(lines[:3], 1):
                        print_color(Color.WHITE, f"    {i}: {line.strip()[:80]}...")
        print()
    
    print_color(Color.MAGENTA, "=" * 70)
    print_color(Color.GREEN if result.is_valid else Color.RED, 
               f"RESULT: {'VALID' if result.is_valid else 'INVALID'}")
    print_color(Color.MAGENTA, "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python debug_validation.py <score_file> <protocol_file>")
        print()
        print("Example:")
        print("  PROTOCOL_SUBSET=dev python scripts/benchmark_py/debug_validation.py \\")
        print("    results/score.txt \\")
        print("    data/protocol.txt")
        sys.exit(1)
    
    score_file = Path(sys.argv[1])
    protocol_file = Path(sys.argv[2])
    
    debug_validation(score_file, protocol_file)
