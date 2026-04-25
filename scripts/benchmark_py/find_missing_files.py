#!/usr/bin/env python3
"""
Find missing files in score file compared to protocol

Usage:
    python find_missing_files.py <score_file> <protocol_file> [protocol_subset]
    
Example:
    python find_missing_files.py \
        logs/results/dataset_score.txt \
        data/dataset/protocol.txt \
        dev
"""

import sys
from pathlib import Path


def read_protocol_ids(protocol_file, subset=None):
    """Read protocol file and extract file IDs"""
    ids = []
    
    with open(protocol_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # If subset specified, filter by subset
            if subset and subset not in line:
                continue
            
            parts = line.split()
            if len(parts) >= 1:
                file_id = parts[0]
                ids.append(file_id)
    
    return ids


def read_score_ids(score_file):
    """Read score file and extract file IDs"""
    ids = []
    
    with open(score_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 1:
                file_id = parts[0]
                ids.append(file_id)
    
    return ids


def main():
    if len(sys.argv) < 3:
        print("Usage: python find_missing_files.py <score_file> <protocol_file> [protocol_subset]")
        print()
        print("Example:")
        print("  python find_missing_files.py \\")
        print("    logs/results/dataset_score.txt \\")
        print("    data/dataset/protocol.txt \\")
        print("    dev")
        sys.exit(1)
    
    score_file = sys.argv[1]
    protocol_file = sys.argv[2]
    protocol_subset = sys.argv[3] if len(sys.argv) > 3 else None
    
    print("═" * 70)
    print("FINDING MISSING FILES")
    print("═" * 70)
    print()
    
    print(f"Score file: {score_file}")
    print(f"Protocol file: {protocol_file}")
    print(f"Protocol subset: {protocol_subset if protocol_subset else 'all'}")
    print()
    
    # Read IDs
    print("Reading protocol IDs...")
    protocol_ids = read_protocol_ids(protocol_file, protocol_subset)
    print(f"  Protocol has {len(protocol_ids)} entries")
    
    print("Reading score IDs...")
    score_ids = read_score_ids(score_file)
    print(f"  Score file has {len(score_ids)} entries")
    
    # Find missing
    protocol_set = set(protocol_ids)
    score_set = set(score_ids)
    missing = protocol_set - score_set
    
    print()
    print(f"Missing: {len(missing)} files ({len(missing)/len(protocol_ids)*100:.1f}%)")
    print()
    
    if missing:
        # Save to file
        output_file = Path(score_file).parent / "missing_files.txt"
        with open(output_file, 'w') as f:
            for file_id in sorted(missing):
                f.write(f"{file_id}\n")
        
        print(f"✓ Saved missing files to: {output_file}")
        print()
        print(f"First 20 missing files:")
        for i, file_id in enumerate(sorted(missing)[:20], 1):
            print(f"  {i:2d}. {file_id}")
        
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
    else:
        print("✓ No missing files!")
    
    print()
    print("═" * 70)


if __name__ == "__main__":
    main()
