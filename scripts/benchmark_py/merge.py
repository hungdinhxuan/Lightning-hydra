"""
Benchmark Merge Module

Handles merged protocol and score file creation.
"""

from pathlib import Path
from typing import List
from datetime import datetime
from .utils import print_color, Color, count_non_empty_lines
from .protocol import extract_eval_subset


def create_merged_protocol(
    results_folder: Path,
    normalized_yaml: str,
    comment: str,
    yaml_config: str,
    base_model_path: str,
    summary_file: Path,
    subdirs: List[Path]
) -> bool:
    """
    Create merged protocol and score files for reuse
    
    Args:
        results_folder: Path to results folder
        normalized_yaml: Normalized YAML config name
        comment: Experiment comment
        yaml_config: YAML config path
        base_model_path: Base model path
        summary_file: Path to summary file
        subdirs: List of subdirectory paths
        
    Returns:
        True if successful, False otherwise
    """
    print_color(Color.CYAN, "🔄 Creating merged protocol and score files for reuse...")
    
    # Define output file paths
    merged_protocol_path = results_folder / f"merged_protocol_{normalized_yaml}_{comment}.txt"
    merged_score_path = results_folder / f"merged_scores_{normalized_yaml}_{comment}.txt"
    metadata_path = results_folder / f"pooled_merged_protocol_{normalized_yaml}_{comment}.txt"
    
    # Remove existing files if they exist
    for path in [merged_protocol_path, merged_score_path, metadata_path]:
        if path.exists():
            path.unlink()
    
    total_entries = 0
    processed_datasets = 0
    dataset_list = []
    dataset_entries_list = []
    
    try:
        # Create metadata file header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(metadata_path, 'w') as f:
            f.write("# Pooled Merged Protocol Metadata\n")
            f.write(f"# Generated: {timestamp}\n")
            f.write(f"# Config: {yaml_config}\n")
            f.write(f"# Base Model: {base_model_path}\n")
            f.write(f"# Comment: {comment}\n")
            f.write("#\n")
            f.write("# Dataset_Name | Entries_Count | Protocol_Path | Score_Path\n")
        
        # Open output files for writing
        with open(merged_protocol_path, 'w') as protocol_out, \
             open(merged_score_path, 'w') as scores_out, \
             open(metadata_path, 'a') as metadata_out:
            
            # Process each subfolder that was successfully processed
            for subfolder in subdirs:
                subfolder_name = subfolder.name
                protocol_path = subfolder / "protocol.txt"
                score_path = results_folder / f"{subfolder_name}_{normalized_yaml}_{comment}.txt"
                
                # Only include datasets that have valid score files
                if not score_path.exists() or not protocol_path.exists():
                    print_color(Color.YELLOW, f"  Skipping {subfolder_name} (missing score or protocol file)")
                    continue
                
                print_color(Color.WHITE, f"  Adding data from: {subfolder_name}")
                
                # Create temporary file for protocol subset
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', prefix='temp_protocol_', suffix='.txt', delete=False) as temp_f:
                    temp_protocol = Path(temp_f.name)
                
                try:
                    # Extract eval subset from protocol file
                    extract_eval_subset(protocol_path, temp_protocol)
                    
                    # Count entries
                    dataset_entries = count_non_empty_lines(temp_protocol)
                    
                    # Modify protocol paths to include dataset name prefix
                    with open(temp_protocol, 'r') as src:
                        for line in src:
                            if line.strip():
                                protocol_out.write(f"{subfolder_name}/{line}")
                    
                    # Process score file - modify paths to include dataset name prefix
                    with open(score_path, 'r') as src:
                        for line in src:
                            if line.strip():
                                scores_out.write(f"{subfolder_name}/{line}")
                    
                    total_entries += dataset_entries
                    processed_datasets += 1
                    dataset_list.append(subfolder_name)
                    dataset_entries_list.append(dataset_entries)
                    
                    # Add dataset info to metadata file
                    metadata_out.write(f"{subfolder_name} | {dataset_entries} | {protocol_path} | {score_path}\n")
                    
                    print_color(Color.WHITE, f"    Added {dataset_entries} entries")
                    
                finally:
                    # Clean up temporary file
                    try:
                        temp_protocol.unlink()
                    except Exception:
                        pass
        
        # Add summary to metadata file
        if total_entries > 0:
            with open(metadata_path, 'a') as f:
                f.write("#\n")
                f.write("# SUMMARY\n")
                f.write(f"TOTAL_DATASETS: {processed_datasets}\n")
                f.write(f"TOTAL_ENTRIES: {total_entries}\n")
                f.write(f"MERGED_PROTOCOL_FILE: {merged_protocol_path}\n")
                f.write(f"MERGED_SCORE_FILE: {merged_score_path}\n")
            
            print_color(Color.GREEN, "✓ Merged files created successfully:")
            print_color(Color.WHITE, f"  Protocol file: {merged_protocol_path}")
            print_color(Color.WHITE, f"  Score file: {merged_score_path}")
            print_color(Color.WHITE, f"  Metadata file: {metadata_path}")
            print_color(Color.WHITE, f"  Total entries: {total_entries}")
            print_color(Color.WHITE, f"  Datasets included: {processed_datasets}")
            
            # Add merged protocol info to summary file
            with open(summary_file, 'a') as f:
                f.write(f"\nMERGED_PROTOCOL: {merged_protocol_path}\n")
                f.write(f"MERGED_SCORES: {merged_score_path}\n")
                f.write(f"PROTOCOL_METADATA: {metadata_path}\n")
                f.write(f"MERGED_ENTRIES: {total_entries}\n")
                f.write(f"MERGED_DATASETS: {processed_datasets}\n")
            
            return True
        else:
            print_color(Color.RED, "❌ Failed to create merged files (no valid entries)")
            return False
            
    except Exception as e:
        print_color(Color.RED, f"❌ Error creating merged files: {e}")
        return False
