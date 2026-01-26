"""
Benchmark EER Calculation Module

Handles pooled and average EER calculations.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional
from .utils import print_color, Color
from .protocol import read_protocol_eval_subset
from .scores import read_scores


def calculate_pooled_eer(
    results_folder: Path,
    normalized_yaml: str,
    comment: str,
    summary_file: Path,
    subdirs: List[Path]
) -> Optional[dict]:
    """
    Calculate pooled EER using dedicated Python script
    
    Args:
        results_folder: Path to results folder
        normalized_yaml: Normalized YAML config name
        comment: Experiment comment
        summary_file: Path to summary file
        subdirs: List of subdirectory paths
        
    Returns:
        Dictionary with pooled EER results or None if failed
    """
    print_color(Color.CYAN, "🔄 Calculating pooled EER using efficient Python implementation...")
    
    # Build command array
    cmd = [
        'python',
        'scripts/calculate_pooled_eer.py',
        str(results_folder),
        normalized_yaml,
        comment
    ]
    
    # Add each benchmark folder as a separate argument
    for subfolder in subdirs:
        cmd.append(str(subfolder))
    
    try:
        # Execute command and capture streams
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        pooled_stdout = result.stdout.strip()
        pooled_stderr = result.stderr
        
        if pooled_stdout:
            # Parse result
            parts = pooled_stdout.split()
            if len(parts) == 5:
                pooled_min_score = parts[0]
                pooled_max_score = parts[1]
                pooled_threshold = parts[2]
                pooled_eer = parts[3]
                pooled_accuracy = parts[4]
                
                # Add pooled EER to summary file
                with open(summary_file, 'a') as f:
                    f.write(f"\nPOOLED_EER | {pooled_eer} | {pooled_min_score} | {pooled_max_score} | {pooled_threshold} | {pooled_accuracy}\n")
                
                # Display the detailed output from Python script (stderr)
                for line in pooled_stderr.splitlines():
                    if "✓" in line:
                        print_color(Color.GREEN, line)
                    elif line.startswith("  "):
                        print_color(Color.WHITE, line)
                    else:
                        print_color(Color.CYAN, line)
                
                return {
                    'min_score': pooled_min_score,
                    'max_score': pooled_max_score,
                    'threshold': pooled_threshold,
                    'eer': pooled_eer,
                    'accuracy': pooled_accuracy
                }
            else:
                print_color(Color.RED, "❌ Invalid result format from pooled EER calculation")
                print_color(Color.YELLOW, f"Output: {pooled_stdout}")
        else:
            print_color(Color.RED, "❌ No output from pooled EER calculation")
            
    except subprocess.CalledProcessError as e:
        print_color(Color.RED, "❌ Failed to calculate pooled EER")
        if e.stderr:
            print_color(Color.YELLOW, f"Error details: {e.stderr}")
    except Exception as e:
        print_color(Color.RED, f"❌ Error: {e}")
    
    return None


def calculate_average_eer(summary_file: Path) -> Optional[float]:
    """
    Calculate average EER across datasets
    
    Args:
        summary_file: Path to summary file
        
    Returns:
        Average EER value or None if failed
    """
    print_color(Color.CYAN, "🔄 Calculating average EER across datasets...")
    
    total_eer = 0.0
    count = 0
    eer_values = []
    
    try:
        # Read individual EER values from summary file
        with open(summary_file, 'r') as f:
            for line in f:
                parts = [p.strip() for p in line.split('|')]
                
                if len(parts) < 2:
                    continue
                
                dataset = parts[0]
                
                # Skip header and empty lines, and exclude pooled/average EER
                if dataset in ['Dataset', 'POOLED_EER', 'AVERAGE_EER', '']:
                    continue
                
                try:
                    eer = float(parts[1])
                    eer_values.append(eer)
                    total_eer += eer
                    count += 1
                except (ValueError, IndexError):
                    continue
        
        if count > 0:
            average_eer = total_eer / count
            
            # Add average EER to summary file
            with open(summary_file, 'a') as f:
                f.write(f"AVERAGE_EER | {average_eer:.6f} | - | - | - | -\n")
            
            # Display average results
            print_color(Color.GREEN, f"✓ Average EER Results (across {count} datasets):")
            print_color(Color.WHITE, f"  Average EER: {average_eer:.6f}")
            print_color(Color.WHITE, f"  Individual EERs: {', '.join(f'{v:.6f}' for v in eer_values)}")
            
            return average_eer
        else:
            print_color(Color.RED, "❌ No valid EER values found for average calculation")
            return None
            
    except Exception as e:
        print_color(Color.RED, f"❌ Error calculating average EER: {e}")
        return None
