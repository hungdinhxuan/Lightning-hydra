#!/usr/bin/env python3
"""
Pooled EER Calculator

Efficiently calculates pooled EER from multiple datasets by combining
all bonafide and spoof samples before computing EER.

Usage:
    python scripts/calculate_pooled_eer.py <results_folder> <normalized_yaml> <comment> <benchmark_folders...>

Arguments:
    results_folder: Path to results folder containing score files
    normalized_yaml: Normalized YAML config name
    comment: Experiment comment
    benchmark_folders: List of benchmark dataset folders
"""

import sys
import os
import subprocess
import tempfile
from pathlib import Path


def read_protocol_eval_subset(protocol_path):
    """Read evaluation subset from protocol file"""
    protocol_entries = []
    
    if not os.path.exists(protocol_path):
        return protocol_entries
    
    try:
        # First pass: check if there are any 'eval' lines
        has_eval_subset = False
        with open(protocol_path, 'r') as f:
            for line in f:
                if 'eval' in line:
                    has_eval_subset = True
                    break
        
        # Second pass: read appropriate lines
        with open(protocol_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 3:  # Need at least filename, subset, label
                    # If there's an eval subset, only process eval lines
                    # If no eval subset, process all lines
                    if not has_eval_subset or 'eval' in line:
                        file_id = parts[0]
                        subset = parts[1]
                        label = parts[2]
                        rest = ' '.join(parts[3:]) if len(parts) > 3 else ''
                        protocol_entries.append((file_id, label, subset))  # Note: label, subset order for compatibility
                        
    except Exception as e:
        print(f"Error reading protocol file {protocol_path}: {e}", file=sys.stderr)
    
    return protocol_entries


def read_scores(score_path):
    """Read scores from score file into dictionary"""
    scores = {}
    
    if not os.path.exists(score_path):
        return scores
    
    try:
        with open(score_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 3:  # Expect 3 columns: filename bonafide_score spoof_score
                    file_id = parts[0]
                    try:
                        bonafide_score = float(parts[1])
                        spoof_score = float(parts[2])
                        scores[file_id] = (bonafide_score, spoof_score)
                    except ValueError:
                        continue
                elif len(parts) >= 2:  # Fallback for 2-column format
                    file_id = parts[0]
                    try:
                        score = float(parts[1])
                        scores[file_id] = (score, score)  # Use same score for both
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading score file {score_path}: {e}", file=sys.stderr)
    
    return scores


def create_combined_files(results_folder, normalized_yaml, comment, benchmark_folders):
    """Create combined protocol and score files from all datasets"""
    
    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    combined_protocol_path = os.path.join(temp_dir, 'combined_protocol.txt')
    combined_scores_path = os.path.join(temp_dir, 'combined_scores.txt')
    
    total_entries = 0
    processed_datasets = 0
    dataset_stats = []
    
    try:
        with open(combined_protocol_path, 'w') as protocol_f, \
             open(combined_scores_path, 'w') as scores_f:
            
            for benchmark_folder in benchmark_folders:
                dataset_name = os.path.basename(benchmark_folder.rstrip('/'))
                
                # Construct file paths
                score_file = os.path.join(results_folder, f"{dataset_name}_{normalized_yaml}_{comment}.txt")
                protocol_file = os.path.join(benchmark_folder, "protocol.txt")
                
                # Debug: Show the files being checked (only if they don't exist)
                if not os.path.exists(score_file) or not os.path.exists(protocol_file):
                    print(f"  Checking dataset: {dataset_name}", file=sys.stderr)
                    print(f"  Score file: {score_file} (exists: {os.path.exists(score_file)})", file=sys.stderr)
                    print(f"  Protocol file: {protocol_file} (exists: {os.path.exists(protocol_file)})", file=sys.stderr)
                
                if not os.path.exists(score_file) or not os.path.exists(protocol_file):
                    print(f"  ⚠️ Skipping {dataset_name} (missing score or protocol file)", file=sys.stderr)
                    continue
                
                print(f"  Processing {dataset_name} for pooled EER...", file=sys.stderr)
                
                # Read protocol and scores
                protocol_entries = read_protocol_eval_subset(protocol_file)
                scores = read_scores(score_file)
                
                dataset_entries = 0
                bonafide_count = 0
                spoof_count = 0
                
                # Process entries
                for file_id, label, subset in protocol_entries:
                    if file_id in scores:
                        # Create unique ID by prefixing with dataset name
                        unique_id = f"{dataset_name}_{file_id}"
                        bonafide_score, spoof_score = scores[file_id]
                        
                        # Write to combined protocol file with proper format: filename subset label
                        protocol_f.write(f"{unique_id} {subset} {label}\n")
                        
                        # Write to combined scores file with proper format: filename bonafide_score spoof_score
                        scores_f.write(f"{unique_id} {bonafide_score} {spoof_score}\n")
                        
                        dataset_entries += 1
                        total_entries += 1
                        
                        if label == 'bonafide':
                            bonafide_count += 1
                        elif label == 'spoof':
                            spoof_count += 1
                
                dataset_stats.append({
                    'name': dataset_name,
                    'entries': dataset_entries,
                    'bonafide': bonafide_count,
                    'spoof': spoof_count
                })
                
                print(f"    Added {dataset_entries} entries ({bonafide_count} bonafide, {spoof_count} spoof)", file=sys.stderr)
                
                # Debug: Show first few labels to understand what's happening
                if dataset_entries > 0 and bonafide_count == 0 and spoof_count == 0:
                    print(f"    ⚠️ No bonafide/spoof labels detected! Checking first few entries:", file=sys.stderr)
                    sample_labels = []
                    for file_id, label, subset in protocol_entries[:5]:  # Show first 5
                        sample_labels.append(f"'{label}'")
                    print(f"    Sample labels: {', '.join(sample_labels)}", file=sys.stderr)
                
                processed_datasets += 1
        
        return combined_protocol_path, combined_scores_path, total_entries, processed_datasets, dataset_stats, temp_dir
        
    except Exception as e:
        # Clean up on error
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise e


def calculate_pooled_eer(results_folder, normalized_yaml, comment, benchmark_folders):
    """Calculate pooled EER using combined temporary files"""
    
    print("🔄 Creating combined protocol and score files for pooled EER...", file=sys.stderr)
    
    # Create combined files
    try:
        combined_protocol_path, combined_scores_path, total_entries, processed_datasets, dataset_stats, temp_dir = create_combined_files(
            results_folder, normalized_yaml, comment, benchmark_folders
        )
    except Exception as e:
        print(f"❌ Error creating combined files: {e}", file=sys.stderr)
        return None
    
    if processed_datasets == 0:
        print("❌ No valid datasets found for pooled EER calculation", file=sys.stderr)
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    
    if total_entries == 0:
        print("❌ No valid entries found for pooled EER calculation", file=sys.stderr)
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    
    print(f"  Total entries: {total_entries}", file=sys.stderr)
    print(f"  Processed datasets: {processed_datasets}", file=sys.stderr)
    
    # Calculate pooled EER using the existing script
    print("🔄 Computing pooled EER using existing evaluation script...", file=sys.stderr)
    
    try:
        # Get the directory of this script to find score_file_to_eer.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        score_script = os.path.join(script_dir, 'score_file_to_eer.py')
        
        result = subprocess.run(
            ['python', score_script, combined_scores_path, combined_protocol_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        pooled_result = result.stdout.strip()
        
        if pooled_result:
            # Parse results
            parts = pooled_result.split()
            if len(parts) >= 5:
                min_score = parts[0]
                max_score = parts[1]
                threshold = parts[2]
                eer = parts[3]
                accuracy = parts[4]
                
                # Output results in the same format as the bash script expects
                print(f"{min_score} {max_score} {threshold} {eer} {accuracy}")
                
                # Also output detailed information to stderr for logging
                print(f"✓ Pooled EER Results (across all {processed_datasets} datasets):", file=sys.stderr)
                print(f"  Pooled EER      : {eer}", file=sys.stderr)
                print(f"  Pooled Accuracy : {accuracy}", file=sys.stderr)
                print(f"  Pooled Threshold: {threshold}", file=sys.stderr)
                print(f"  Min Score       : {min_score}", file=sys.stderr)
                print(f"  Max Score       : {max_score}", file=sys.stderr)
                print(f"  Total Samples   : {total_entries}", file=sys.stderr)
                
                return {
                    'min_score': min_score,
                    'max_score': max_score,
                    'threshold': threshold,
                    'eer': eer,
                    'accuracy': accuracy,
                    'total_entries': total_entries,
                    'processed_datasets': processed_datasets,
                    'dataset_stats': dataset_stats
                }
            else:
                print("❌ Invalid result format from score calculation", file=sys.stderr)
                return None
        else:
            print("❌ No result from score calculation", file=sys.stderr)
            return None
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error calculating pooled EER: {e}", file=sys.stderr)
        print(f"   stderr: {e.stderr}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        return None
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    if len(sys.argv) < 5:
        print("Usage: python scripts/calculate_pooled_eer.py <results_folder> <normalized_yaml> <comment> <benchmark_folder1> [benchmark_folder2] ...", file=sys.stderr)
        sys.exit(1)
    
    results_folder = sys.argv[1]
    normalized_yaml = sys.argv[2]
    comment = sys.argv[3]
    benchmark_folders = sys.argv[4:]
    
    result = calculate_pooled_eer(results_folder, normalized_yaml, comment, benchmark_folders)
    
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main() 