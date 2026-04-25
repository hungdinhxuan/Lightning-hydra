"""
Benchmark EER Calculation Module

Handles pooled and average EER calculations.
"""

import json
import subprocess
from pathlib import Path
from typing import List, Optional
from .utils import print_color, Color


def _safe_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_float(value) -> str:
    return f"{_safe_float(value):.6f}"


def _format_percent_from_fraction(value) -> str:
    return f"{_safe_float(value) * 100.0:.6f}"


def calculate_pooled_eer(
    results_folder: Path,
    normalized_yaml: str,
    comment: str,
    summary_file: Path,
    subdirs: List[Path],
    eval_config_path: Optional[Path] = None,
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
        result = subprocess.run(
            cmd + (['--eval-config', str(eval_config_path)] if eval_config_path else []) + ['--output-format', 'json'],
            capture_output=True,
            text=True,
            check=True
        )
        
        pooled_stdout = result.stdout.strip()
        
        if pooled_stdout:
            payload = json.loads(pooled_stdout)
            legacy = payload['legacy_compat']
            detailed_results = payload['results']

            metadata_lines = []
            if summary_file.exists():
                with open(summary_file, 'r') as handle:
                    for line in handle:
                        stripped = line.strip()
                        if not stripped:
                            break
                        metadata_lines.append(stripped)

            per_dataset_results = detailed_results['threshold_free']['per_dataset']
            with open(summary_file, 'w') as handle:
                if metadata_lines:
                    handle.write("\n".join(metadata_lines))
                    handle.write("\n\n")
                handle.write("Dataset | EER | ROC-AUC | Accuracy\n")
                for record in per_dataset_results:
                    dataset_accuracy = record.get('legacy_accuracy')
                    handle.write(
                        f"{record['dataset_name']} | "
                        f"{_format_percent_from_fraction(record['threshold_free'].get('eer'))} | "
                        f"{_format_percent_from_fraction(record['threshold_free'].get('roc_auc'))} | "
                        f"{_format_float(dataset_accuracy)}\n"
                    )
                handle.write("\n")
                handle.write(
                    f"POOLED_EER | "
                    f"{_format_float(legacy.get('eer'))} | "
                    f"{_format_percent_from_fraction(detailed_results['threshold_free']['raw_pooled']['threshold_free'].get('roc_auc'))} | "
                    f"{_format_float(legacy.get('accuracy'))}\n"
                )
                handle.write(
                    f"BALANCED_POOLED_EER | "
                    f"{_format_percent_from_fraction(detailed_results['threshold_free']['balanced_pooled']['threshold_free'].get('eer'))} | "
                    f"{_format_percent_from_fraction(detailed_results['threshold_free']['balanced_pooled']['threshold_free'].get('roc_auc'))} | "
                    f"{_format_percent_from_fraction(detailed_results['threshold_based'].get('eer_threshold', {}).get('balanced_pooled', {}).get('accuracy'))}\n"
                )
                handle.write(
                    f"AVERAGE_EER | "
                    f"{_format_percent_from_fraction(detailed_results['threshold_free'].get('macro_average_eer'))} | - | -\n"
                )

            detailed_summary_file = summary_file.with_name(f"{summary_file.stem}_detailed.txt")
            with open(detailed_summary_file, 'w') as handle:
                handle.write("## Pooled Evaluation\n")
                handle.write(detailed_results['summary_text'])
                handle.write("\n\n")

            detailed_jsonl_file = summary_file.with_name(f"{summary_file.stem}_details.jsonl")
            with open(detailed_jsonl_file, 'w') as handle:
                handle.write(json.dumps({
                    'type': 'pooled',
                    'payload': payload,
                }))
                handle.write('\n')

            raw_pooled = detailed_results['threshold_free']['raw_pooled']['threshold_free']
            balanced_pooled = detailed_results['threshold_free']['balanced_pooled']['threshold_free']

            print_color(Color.GREEN, "✓ Pooled EER Results:")
            print_color(Color.WHITE, f"  Raw pooled EER      : {_format_float(legacy.get('eer'))}")
            print_color(Color.WHITE, f"  Raw pooled Accuracy : {_format_float(legacy.get('accuracy'))}")
            print_color(Color.WHITE, f"  Raw pooled ROC-AUC  : {_format_percent_from_fraction(raw_pooled.get('roc_auc'))}")
            print_color(Color.WHITE, f"  Balanced pooled EER : {_format_percent_from_fraction(balanced_pooled.get('eer'))}")
            print_color(Color.WHITE, f"  Macro-average EER   : {_format_percent_from_fraction(detailed_results['threshold_free'].get('macro_average_eer'))}")
            print_color(Color.WHITE, f"  Detailed summary    : {detailed_summary_file}")

            return {
                'min_score': _safe_float(legacy.get('min_score')),
                'max_score': _safe_float(legacy.get('max_score')),
                'threshold': _safe_float(legacy.get('threshold')),
                'eer': _safe_float(legacy.get('eer')),
                'accuracy': _safe_float(legacy.get('accuracy')),
                'balanced_pooled_eer': _safe_float(balanced_pooled.get('eer')) * 100.0,
                'macro_average_eer': _safe_float(detailed_results['threshold_free'].get('macro_average_eer')) * 100.0,
            }
        else:
            print_color(Color.RED, "❌ No output from pooled EER calculation")
            
    except subprocess.CalledProcessError as e:
        print_color(Color.RED, "❌ Failed to calculate pooled EER")
        if e.stderr:
            print_color(Color.YELLOW, f"Error details: {e.stderr}")
    except json.JSONDecodeError as e:
        print_color(Color.RED, "❌ Invalid JSON from pooled EER calculation")
        print_color(Color.YELLOW, f"Error details: {e}")
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
                if dataset in ['Dataset', 'POOLED_EER', 'BALANCED_POOLED_EER', 'AVERAGE_EER', '']:
                    continue
                
                try:
                    eer = float(parts[1])
                    if eer != eer:
                        continue
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
