#!/usr/bin/env python3
"""
Example usage of the create_softlinks_from_protocol function
"""

from create_softlinks import create_softlinks_from_protocol
from pathlib import Path


def example_usage():
    """Demonstrate how to use the function programmatically"""
    
    # Define paths
    protocol_file = "/home/hungdx/code/Lightning-hydra/data/spoofceleb/protocol.txt"
    source_base_dir = "/home/hungdx/code/Lightning-hydra/data/spoofceleb"
    target_base_dir = "/home/hungdx/code/Lightning-hydra/data/spoofceleb_filtered"
    
    # Example 1: Dry run to see what would be created
    print("=" * 80)
    print("Example 1: Dry run (train and dev sets, bonafide only, with 'a00')")
    print("=" * 80)
    stats = create_softlinks_from_protocol(
        protocol_file=protocol_file,
        source_base_dir=source_base_dir,
        target_base_dir=target_base_dir,
        sets_to_include=("train", "dev"),
        labels_to_include=("bonafide",),
        path_filter="a00",
        dry_run=True
    )
    print(f"\nStats: {stats}\n")
    
    
    # Example 2: Only train set, bonafide only, with 'a00'
    print("=" * 80)
    print("Example 2: Only train set, bonafide only, with 'a00' (dry run)")
    print("=" * 80)
    stats = create_softlinks_from_protocol(
        protocol_file=protocol_file,
        source_base_dir=source_base_dir,
        target_base_dir=target_base_dir + "_train_only",
        sets_to_include=("train",),
        labels_to_include=("bonafide",),
        path_filter="a00",
        dry_run=True
    )
    print(f"\nStats: {stats}\n")
    
    
    # Example 3: To actually create links, set dry_run=False
    # Uncomment the following to actually create the links
    """
    print("=" * 80)
    print("Example 3: Actually create the links")
    print("=" * 80)
    stats = create_softlinks_from_protocol(
        protocol_file=protocol_file,
        source_base_dir=source_base_dir,
        target_base_dir=target_base_dir,
        sets_to_include=("train", "dev"),
        labels_to_include=("bonafide",),
        path_filter="a00",
        dry_run=False  # This will actually create the links
    )
    print(f"\nStats: {stats}\n")
    """


if __name__ == "__main__":
    example_usage()

