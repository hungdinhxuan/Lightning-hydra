#!/usr/bin/env python3
"""
Script to create softlinks for audio files based on protocol.txt
Filters for train/dev sets and files containing 'a00' in the path
"""

import os
import sys
from pathlib import Path
from typing import Optional


def create_softlinks_from_protocol(
    protocol_file: str,
    source_base_dir: str,
    target_base_dir: str,
    sets_to_include: tuple = ("train", "dev"),
    labels_to_include: tuple = ("bonafide",),
    path_filter: str = "a00",
    dry_run: bool = False
) -> dict:
    """
    Create softlinks maintaining the same directory structure as protocol file.
    
    Args:
        protocol_file: Path to the protocol.txt file
        source_base_dir: Base directory where the actual audio files are located
        target_base_dir: Base directory where softlinks will be created
        sets_to_include: Tuple of dataset splits to include (default: train and dev)
        labels_to_include: Tuple of labels to include (default: bonafide only)
        path_filter: String that must be present in the file path (default: "a00")
        dry_run: If True, only print what would be done without creating links
    
    Returns:
        Dictionary with statistics about the operation
    """
    
    stats = {
        "total_lines": 0,
        "filtered_lines": 0,
        "links_created": 0,
        "links_skipped": 0,
        "errors": 0
    }
    
    protocol_path = Path(protocol_file)
    source_base = Path(source_base_dir)
    target_base = Path(target_base_dir)
    
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_file}")
    
    if not source_base.exists():
        raise FileNotFoundError(f"Source base directory not found: {source_base_dir}")
    
    print(f"Reading protocol file: {protocol_file}")
    print(f"Source base directory: {source_base_dir}")
    print(f"Target base directory: {target_base_dir}")
    print(f"Sets to include: {sets_to_include}")
    print(f"Labels to include: {labels_to_include}")
    print(f"Path filter: '{path_filter}'")
    print(f"Dry run: {dry_run}")
    print("-" * 80)
    
    with open(protocol_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            stats["total_lines"] += 1
            
            # Parse line: format is "filepath set label"
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            
            file_path, set_name, label = parts[0], parts[1], parts[2]
            
            # Filter by set (train/dev)
            if set_name not in sets_to_include:
                continue
            
            # Filter by label (bonafide/spoof)
            if label not in labels_to_include:
                continue
            
            # Filter by path (must contain 'a00')
            if path_filter and path_filter not in file_path:
                continue
            
            stats["filtered_lines"] += 1
            
            # Build source and target paths
            source_file = source_base / file_path
            target_file = target_base / file_path
            
            # Check if source file exists
            if not source_file.exists():
                print(f"WARNING: Source file not found: {source_file}")
                stats["errors"] += 1
                continue
            
            # Check if target already exists
            if target_file.exists() or target_file.is_symlink():
                stats["links_skipped"] += 1
                if stats["links_skipped"] <= 5:  # Only print first few
                    print(f"SKIP: Target already exists: {target_file}")
                continue
            
            if dry_run:
                print(f"WOULD CREATE: {target_file} -> {source_file}")
                stats["links_created"] += 1
            else:
                try:
                    # Create parent directories if they don't exist
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Create symbolic link
                    os.symlink(source_file, target_file)
                    stats["links_created"] += 1
                    
                    if stats["links_created"] <= 10:  # Print first few
                        print(f"CREATED: {target_file} -> {source_file}")
                    elif stats["links_created"] % 1000 == 0:  # Print progress
                        print(f"Progress: {stats['links_created']} links created...")
                        
                except Exception as e:
                    print(f"ERROR creating link {target_file}: {e}")
                    stats["errors"] += 1
            
            # Print progress for large files
            if line_num % 100000 == 0:
                print(f"Processed {line_num:,} lines...")
    
    # Print summary
    print("-" * 80)
    print("Summary:")
    print(f"  Total lines processed: {stats['total_lines']:,}")
    print(f"  Lines matching filters: {stats['filtered_lines']:,}")
    print(f"  Links created: {stats['links_created']:,}")
    print(f"  Links skipped (already exist): {stats['links_skipped']:,}")
    print(f"  Errors: {stats['errors']:,}")
    
    return stats


def main():
    """Main function with example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create softlinks for audio files based on protocol.txt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Dry run to see what would be created
  python create_softlinks.py \\
      --protocol data/spoofceleb/protocol.txt \\
      --source-base data/spoofceleb \\
      --target-base data/spoofceleb_filtered \\
      --dry-run
  
  # Actually create the links
  python create_softlinks.py \\
      --protocol data/spoofceleb/protocol.txt \\
      --source-base data/spoofceleb \\
      --target-base data/spoofceleb_filtered
        """
    )
    
    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        help="Path to protocol.txt file"
    )
    parser.add_argument(
        "--source-base",
        type=str,
        required=True,
        help="Base directory where actual audio files are located"
    )
    parser.add_argument(
        "--target-base",
        type=str,
        required=True,
        help="Base directory where softlinks will be created"
    )
    parser.add_argument(
        "--sets",
        type=str,
        nargs="+",
        default=["train", "dev"],
        help="Dataset splits to include (default: train dev)"
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=["bonafide"],
        help="Labels to include (default: bonafide)"
    )
    parser.add_argument(
        "--path-filter",
        type=str,
        default="a00",
        help="String that must be present in file path (default: a00)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without creating links"
    )
    
    args = parser.parse_args()
    
    try:
        create_softlinks_from_protocol(
            protocol_file=args.protocol,
            source_base_dir=args.source_base,
            target_base_dir=args.target_base,
            sets_to_include=tuple(args.sets),
            labels_to_include=tuple(args.labels),
            path_filter=args.path_filter,
            dry_run=args.dry_run
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

