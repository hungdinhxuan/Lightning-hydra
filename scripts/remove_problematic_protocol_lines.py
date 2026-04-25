#!/usr/bin/env python3
"""
Script to remove problematic lines from protocol files based on problem_files.log

The script:
1. Parses problem_files.log to extract problematic file paths
2. Deletes the problematic audio files from the filesystem
3. Removes corresponding lines from all protocol_*.txt files
4. Creates cleaned versions of the protocol files
"""

import re
from pathlib import Path
from typing import Set


def parse_problem_files(problem_log_path: str) -> Set[str]:
    """
    Parse problem_files.log to extract problematic file paths.
    
    Args:
        problem_log_path: Path to problem_files.log
        
    Returns:
        Set of problematic file paths (without prefix/suffix formatting)
    """
    problematic_paths = set()
    
    with open(problem_log_path, 'r') as f:
        for line in f:
            # Match lines like: "Line X: subset=eval label=bonafide path=..."
            match = re.search(r'path=([^\s]+)', line)
            if match:
                path = match.group(1)
                problematic_paths.add(path)
    
    return problematic_paths


def delete_problematic_files(
    data_dir: Path,
    problematic_paths: Set[str]
) -> dict[str, bool]:
    """
    Delete problematic audio files from the filesystem.
    
    Args:
        data_dir: Base data directory (e.g., /path/to/data/MLAAD)
        problematic_paths: Set of problematic paths (relative to data_dir)
        
    Returns:
        Dictionary mapping file paths to deletion status (True=deleted, False=not found or error)
    """
    deletion_results = {}
    
    print("=" * 80)
    print("Deleting problematic audio files...")
    print("=" * 80)
    
    for problematic_path in sorted(problematic_paths):
        # Construct full path: data_dir / problematic_path
        full_file_path = data_dir / problematic_path
        
        deletion_results[str(full_file_path)] = False
        
        if full_file_path.exists():
            try:
                full_file_path.unlink()  # Delete the file
                deletion_results[str(full_file_path)] = True
                print(f"  ✓ Deleted: {problematic_path}")
            except Exception as e:
                print(f"  ✗ Error deleting {problematic_path}: {e}")
        else:
            print(f"  ! Not found (skipping): {problematic_path}")
    
    deleted_count = sum(deletion_results.values())
    print(f"\nDeleted {deleted_count} out of {len(problematic_paths)} problematic files.\n")
    
    return deletion_results


def remove_problematic_lines(
    protocol_file_path: str, 
    problematic_paths: Set[str],
    output_path: str = None
) -> tuple[int, int]:
    """
    Remove lines containing problematic paths from a protocol file.
    
    Args:
        protocol_file_path: Path to the protocol file to clean
        problematic_paths: Set of problematic paths to remove
        output_path: Optional output path (if None, overwrites original)
        
    Returns:
        Tuple of (total_lines_before, removed_lines_count)
    """
    if output_path is None:
        output_path = protocol_file_path
    
    removed_count = 0
    total_lines = 0
    kept_lines = []
    
    print(f"Processing {protocol_file_path}...")
    
    with open(protocol_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.rstrip('\n\r')
            
            # Protocol files have format: "path subset label"
            # Check if this line starts with any problematic path
            should_remove = False
            for problematic_path in problematic_paths:
                # Match if line starts with the problematic path followed by a space
                if line.startswith(problematic_path + ' ') or line.strip() == problematic_path:
                    should_remove = True
                    removed_count += 1
                    print(f"  Removing line {line_num}: {line[:80]}...")
                    break
            
            if not should_remove:
                kept_lines.append(line)
    
    # Write cleaned file
    print(f"  Writing cleaned file to {output_path}...")
    with open(output_path, 'w') as f:
        for line in kept_lines:
            f.write(line + '\n')
    
    print(f"  Removed {removed_count} lines out of {total_lines} total lines.")
    print(f"  Kept {len(kept_lines)} lines.\n")
    
    return total_lines, removed_count


def main():
    """Main function to clean all protocol files."""
    # Paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "MLAAD"
    problem_log_path = data_dir / "problem_files.log"
    
    # Protocol files to process
    protocol_files = [
        "protocol_v6.txt",
        "protocol_v7.txt", 
        "protocol_v8.txt",
        "protocol_dev_v9.txt"
    ]
    
    # Parse problematic paths
    print("=" * 80)
    print("Parsing problem_files.log...")
    print("=" * 80)
    problematic_paths = parse_problem_files(str(problem_log_path))
    print(f"Found {len(problematic_paths)} problematic paths:\n")
    for path in sorted(problematic_paths):
        print(f"  - {path}")
    print()
    
    # Delete problematic audio files
    deletion_results = delete_problematic_files(data_dir, problematic_paths)
    
    # Process each protocol file
    print("=" * 80)
    print("Processing protocol files...")
    print("=" * 80)
    
    results = {}
    for protocol_file in protocol_files:
        protocol_path = data_dir / protocol_file
        if not protocol_path.exists():
            print(f"Warning: {protocol_file} not found, skipping...\n")
            continue
        
        total_lines, removed = remove_problematic_lines(
            str(protocol_path),
            problematic_paths
        )
        results[protocol_file] = {
            'total': total_lines,
            'removed': removed,
            'kept': total_lines - removed
        }
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    for protocol_file, stats in results.items():
        print(f"{protocol_file}:")
        print(f"  Total lines: {stats['total']}")
        print(f"  Removed: {stats['removed']}")
        print(f"  Kept: {stats['kept']}")
        print()


if __name__ == "__main__":
    main()

