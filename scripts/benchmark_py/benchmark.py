#!/usr/bin/env python3
"""
Bulk Benchmark Runner - Python Implementation

A Python refactor of the bash benchmark scripts for easier maintenance and extensibility.

Usage:
    python benchmark.py -g GPU -c CONFIG -b BENCHMARK_DIR -m MODEL_PATH -r RESULTS_DIR -n COMMENT [OPTIONS]
"""

import argparse
import sys
import os
import random
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_py.constants import CONSTANTS
from benchmark_py.utils import (
    print_color, Color, print_banner, print_usage,
    display_progress, cleanup_temp_files, normalize_yaml_name
)
from benchmark_py.validation import validate_score_file, check_protocol_exists
from benchmark_py.protocol import create_missing_protocol
from benchmark_py.scores import merge_score_files
from benchmark_py.execution import BenchmarkConfig, execute_benchmark, evaluate_results
from benchmark_py.eer import calculate_pooled_eer
from benchmark_py.merge import create_merged_protocol


def infer_eval_config_path(benchmark_folder: Path) -> Optional[Path]:
    candidate = Path("configs/eval") / f"{benchmark_folder.name}.yaml"
    return candidate if candidate.exists() else None


def resolve_eval_config_path(
    benchmark_folder: Path,
    explicit_eval_config: Optional[Path],
) -> Optional[Path]:
    if explicit_eval_config is not None:
        return explicit_eval_config
    return infer_eval_config_path(benchmark_folder)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Bulk Benchmark Runner Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=print_usage()
    )
    
    # Required arguments
    parser.add_argument('-g', '--gpu', required=True,
                       help='GPU identifier (index like 0/1 or MIG UUID)')
    parser.add_argument('-c', '--config', required=True,
                       help='YAML config file path (e.g., cnsl/xlsr_vib_large_corpus)')
    parser.add_argument('-b', '--benchmark-folder', required=True, type=Path,
                       help='Bulk benchmark folder path')
    parser.add_argument('-m', '--model-path', required=True,
                       help='Base model path')
    parser.add_argument('-r', '--results-folder', required=True, type=Path,
                       help='Results folder path')
    parser.add_argument('-n', '--comment', required=True,
                       help='Comment to note')
    
    # Optional arguments
    parser.add_argument('-a', '--adapter-paths', default=None,
                       help='Adapter paths (optional)')
    parser.add_argument('-l', '--is-ln', default=CONSTANTS.default_is_base_model_path_ln,
                       type=lambda x: x.lower() == 'true',
                       help='Whether to use Lightning checkpoint loading (default: true)')
    parser.add_argument('-s', '--random-start', default=CONSTANTS.default_is_random_start,
                       type=lambda x: x.lower() == 'true',
                       help='Whether to use random start (default: true)')
    parser.add_argument('-t', '--trim-length', default=CONSTANTS.default_trim_length,
                       type=int,
                       help='Trim length for data processing (default: 64000)')
    parser.add_argument('-z', '--batch-size', default=CONSTANTS.default_batch_size,
                       type=int,
                       help='Batch size (default: 128)')
    parser.add_argument('--eval-config', default=None, type=Path,
                       help='Optional eval yaml path for fill_policy and reporting. Default: auto-detect configs/eval/<benchmark_folder_name>.yaml')
    
    args, unknown_args = parser.parse_known_args()
    args.extra_overrides = unknown_args
    return args


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate required arguments
    
    Args:
        args: Parsed arguments
        
    Returns:
        True if valid, False otherwise
    """
    invalid_overrides = [
        arg for arg in args.extra_overrides
        if not (arg.startswith("+") or arg.startswith("++"))
    ]
    if invalid_overrides:
        print_color(
            Color.RED,
            "Error: Unknown non-Hydra arguments found: "
            + ", ".join(invalid_overrides)
        )
        print_color(
            Color.YELLOW,
            "Hint: Extra arguments must be Hydra overrides, e.g. +trainer.precision=bf16-mixed"
        )
        return False

    if not args.benchmark_folder.exists():
        print_color(Color.RED, f"Error: Benchmark folder '{args.benchmark_folder}' does not exist.")
        return False
    
    if not args.benchmark_folder.is_dir():
        print_color(Color.RED, f"Error: Benchmark folder '{args.benchmark_folder}' is not a directory.")
        return False

    if args.eval_config is not None and not args.eval_config.exists():
        print_color(Color.RED, f"Error: Eval config '{args.eval_config}' does not exist.")
        return False

    if args.eval_config is not None and not args.eval_config.is_file():
        print_color(Color.RED, f"Error: Eval config '{args.eval_config}' is not a file.")
        return False
    
    return True


def initialize_results(
    results_folder: Path,
    comment: str,
    yaml_config: str,
    base_model_path: str,
    adapter_paths: Optional[str],
    is_ln: bool,
    trim_length: int,
    eval_config_path: Optional[Path],
) -> tuple:
    """
    Initialize results directory and summary file
    
    Args:
        results_folder: Base results folder path
        comment: Experiment comment
        yaml_config: YAML config path
        base_model_path: Base model path
        adapter_paths: Adapter paths (optional)
        is_ln: Whether Lightning checkpoint loading is enabled
        trim_length: Trim length value
        
    Returns:
        Tuple of (complete_results_folder, summary_file, normalized_yaml)
    """
    # Create complete results directory with comment subfolder
    complete_results_folder = results_folder / comment
    complete_results_folder.mkdir(parents=True, exist_ok=True)
    
    # Create summary file
    summary_file = complete_results_folder / CONSTANTS.summary_file_name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Normalize YAML config for file naming
    normalized_yaml = normalize_yaml_name(yaml_config)
    
    # Write header to summary file
    with open(summary_file, 'w') as f:
        f.write(f"Config: {yaml_config}\n")
        f.write(f"Base_model_path: {base_model_path}\n")
        f.write(f"Lora Path: {adapter_paths if adapter_paths else 'None'}\n")
        f.write(f"Is Base Model Path LN: {is_ln}\n")
        f.write(f"Trim Length: {trim_length}\n")
        f.write(f"Eval Config: {eval_config_path if eval_config_path else 'None'}\n")
        f.write(f"Date: {timestamp}\n")
        f.write("\n")
        f.write("Dataset | EER | ROC-AUC | Accuracy\n")
    
    return complete_results_folder, summary_file, normalized_yaml


def get_subdirectories(benchmark_folder: Path) -> List[Path]:
    """
    Get list of subdirectories in benchmark folder
    
    Args:
        benchmark_folder: Path to benchmark folder
        
    Returns:
        List of subdirectory paths
    """
    subdirs = []
    
    try:
        for item in sorted(benchmark_folder.iterdir()):
            if item.is_dir():
                print_color(Color.WHITE, f"Found directory: {item.name}")
                subdirs.append(item)
    except Exception as e:
        print_color(Color.RED, f"Error reading benchmark folder: {e}")
    
    return subdirs


def process_dataset(
    subfolder: Path,
    gpu_number: str,
    yaml_config: str,
    base_model_path: str,
    results_folder: Path,
    normalized_yaml: str,
    comment: str,
    adapter_paths: Optional[str],
    is_ln: bool,
    is_random_start: bool,
    trim_length: int,
    batch_size: int,
    summary_file: Path,
    extra_overrides: List[str],
    eval_config_path: Optional[Path],
    benchmark_root: Path,
) -> bool:
    """
    Process a single dataset
    
    Args:
        subfolder: Path to dataset subfolder
        gpu_number: GPU identifier
        yaml_config: YAML config path
        base_model_path: Base model path
        results_folder: Results folder path
        normalized_yaml: Normalized YAML config name
        comment: Experiment comment
        adapter_paths: Adapter paths (optional)
        is_ln: Whether Lightning checkpoint loading is enabled
        is_random_start: Whether random start is enabled
        trim_length: Trim length value
        batch_size: Batch size value
        summary_file: Path to summary file
        extra_overrides: Additional Hydra overrides passed from CLI
        
    Returns:
        True if successful, False otherwise
    """
    subfolder_name = subfolder.name
    
    # Set paths
    data_dir = subfolder
    protocol_path = subfolder / "protocol.txt"
    score_save_path = results_folder / f"{subfolder_name}_{normalized_yaml}_{comment}.txt"
    
    # Initialize variables for protocol and score path handling
    protocol_to_use = protocol_path
    score_path_to_use = score_save_path
    use_temp_protocol = False
    temp_protocol_path = None
    temp_score_path = None
    
    # Check if score file exists and is complete
    validation_result = validate_score_file(score_save_path, protocol_path)
    
    if validation_result.is_valid:
        # Score file is complete, just evaluate
        if evaluate_results(score_save_path, protocol_path, summary_file, subfolder_name, eval_config_path, results_folder, benchmark_root, normalized_yaml, comment):
            print_color(Color.GREEN, f"✓ Results for {subfolder_name} (using existing complete score file)")
            return True
        else:
            print_color(Color.RED, f"❌ Error: Failed to evaluate results for {subfolder_name}")
            return False
    
    elif score_save_path.exists():
        # Score file exists but is incomplete
        print_color(Color.YELLOW, f"⚠️ Warning: Score file exists but is incomplete/corrupted for {subfolder_name}.")
        print_color(Color.WHITE, f"  Current protocol subset: {CONSTANTS.protocol_subset}")
        
        # Create temporary protocol file with missing entries
        random_id = f"{os.getpid()}_{int(time.time())}_{random.randint(0, 99999)}"
        temp_protocol_path = results_folder / f"{CONSTANTS.temp_protocol_prefix}_{subfolder_name}_{random_id}.txt"
        
        try:
            missing_count = create_missing_protocol(score_save_path, protocol_path, temp_protocol_path)
            
            if missing_count > 0:
                print_color(Color.CYAN, f"🔄 Running benchmark for {missing_count} missing entries only...")
                print_color(Color.WHITE, f"  Temp protocol: {temp_protocol_path}")
                protocol_to_use = temp_protocol_path
                temp_score_path = results_folder / f"{CONSTANTS.temp_scores_prefix}_{subfolder_name}_{random_id}.txt"
                score_path_to_use = temp_score_path
                use_temp_protocol = True
            elif missing_count == 0:
                print_color(Color.GREEN, "✓ No missing entries found. Score file is actually complete.")
                # Re-validate to make sure
                validation_result = validate_score_file(score_save_path, protocol_path, verbose=False)
                if validation_result.is_valid:
                    if evaluate_results(score_save_path, protocol_path, summary_file, subfolder_name, eval_config_path, results_folder, benchmark_root, normalized_yaml, comment):
                        print_color(Color.GREEN, f"✓ Results for {subfolder_name} (using existing complete score file)")
                return True
            else:
                print_color(Color.RED, "❌ Error: Failed to analyze missing entries. Re-running full benchmark...")
                protocol_to_use = protocol_path
                score_path_to_use = score_save_path
                use_temp_protocol = False
        except Exception as e:
            print_color(Color.RED, f"❌ Error creating missing protocol: {e}")
            print_color(Color.YELLOW, "  Falling back to full benchmark re-run...")
            protocol_to_use = protocol_path
            score_path_to_use = score_save_path
            use_temp_protocol = False
    
    else:
        # No existing score file
        print_color(Color.CYAN, f"ℹ️ No existing score file found for {subfolder_name}. Running fresh benchmark...")
        protocol_to_use = protocol_path
        score_path_to_use = score_save_path
        use_temp_protocol = False
    
    # Check if protocol file exists
    if not check_protocol_exists(protocol_to_use):
        print_color(Color.RED, f"⚠️ Warning: Protocol file not found at {protocol_to_use}. Skipping this dataset.")
        return False
    
    # Debug: Show protocol file content summary
    if use_temp_protocol and protocol_to_use.exists():
        try:
            with open(protocol_to_use, 'r') as f:
                temp_protocol_lines = sum(1 for line in f if line.strip() and not line.startswith('#'))
            print_color(Color.WHITE, f"  Temp protocol has {temp_protocol_lines} lines to process")
            if temp_protocol_lines == 0:
                print_color(Color.RED, f"  ⚠️ Warning: Temp protocol file is empty!")
        except Exception as e:
            print_color(Color.YELLOW, f"  Could not read temp protocol: {e}")
    
    # Construct benchmark configuration
    config = BenchmarkConfig(
        gpu_number=gpu_number,
        yaml_config=yaml_config,
        score_save_path=score_path_to_use,
        data_dir=data_dir,
        protocol_path=protocol_to_use,
        base_model_path=base_model_path,
        is_base_model_path_ln=is_ln,
        is_random_start=is_random_start,
        trim_length=trim_length,
        batch_size=batch_size,
        adapter_paths=adapter_paths,
        extra_overrides=extra_overrides
    )
    
    # Debug: Show what paths we're using
    if use_temp_protocol:
        print_color(Color.WHITE, f"  Using temp protocol: {protocol_to_use}")
        print_color(Color.WHITE, f"  Expecting temp scores at: {score_path_to_use}")
    
    # Record files before benchmark
    before_time = time.time()
    
    # Execute benchmark
    if not execute_benchmark(config):
        print_color(Color.RED, f"❌ Benchmark failed for {subfolder_name}")
        return False
    
    # Debug: Check what files were created/modified after benchmark
    print_color(Color.CYAN, "🔍 Checking for new/modified files after benchmark...")
    try:
        new_files = []
        for file in results_folder.glob("*"):
            if file.is_file() and file.stat().st_mtime > before_time:
                new_files.append(file)
        
        if new_files:
            print_color(Color.GREEN, f"  ✓ Found {len(new_files)} new/modified file(s):")
            for file in new_files:
                print_color(Color.WHITE, f"    - {file.name}")
        else:
            print_color(Color.YELLOW, f"  ⚠️ No new files were created in {results_folder}")
    except Exception as e:
        print_color(Color.YELLOW, f"  Could not check for new files: {e}")
    
    # Debug: Check if temp score file was created
    if use_temp_protocol:
        if temp_score_path and temp_score_path.exists():
            print_color(Color.GREEN, f"  ✓ Temp score file was created: {temp_score_path}")
            # Show file size
            try:
                size = temp_score_path.stat().st_size
                print_color(Color.WHITE, f"    File size: {size} bytes")
            except:
                pass
        else:
            print_color(Color.RED, f"  ❌ Temp score file was NOT created: {temp_score_path}")
            # Check if original score file was modified instead
            if score_save_path.exists():
                try:
                    mtime = score_save_path.stat().st_mtime
                    if mtime > before_time:
                        print_color(Color.YELLOW, f"  ⚠️ Warning: Original score file was modified instead!")
                        print_color(Color.YELLOW, f"    This means the benchmark wrote to wrong path")
                except:
                    pass
    
    # Handle temporary protocol case - merge results if needed
    if use_temp_protocol:
        if temp_score_path and temp_score_path.exists():
            # Count temp score lines
            temp_score_lines = 0
            try:
                with open(temp_score_path, 'r') as f:
                    for line in f:
                        if line.strip() and not line.startswith('#'):
                            temp_score_lines += 1
                print_color(Color.WHITE, f"  Temp score file has {temp_score_lines} lines")
            except Exception as e:
                print_color(Color.YELLOW, f"  Could not count temp score lines: {e}")
            
            print_color(Color.CYAN, "🔄 Merging temporary scores with existing scores...")
            merged_score_path = results_folder / f"{CONSTANTS.merged_scores_prefix}_{subfolder_name}_{random_id}.txt"
            
            # Merge the scores
            if merge_score_files(score_save_path, temp_score_path, merged_score_path):
                print_color(Color.GREEN, "✓ Score files merged successfully")
                
                # Count merged lines
                try:
                    merged_lines = 0
                    with open(score_save_path, 'r') as f:
                        for line in f:
                            if line.strip() and not line.startswith('#'):
                                merged_lines += 1
                    print_color(Color.WHITE, f"  After merge: {merged_lines} total lines")
                except Exception:
                    pass
            else:
                print_color(Color.RED, "❌ Failed to merge score files")
            
            # Clean up temporary files
            try:
                if temp_protocol_path:
                    temp_protocol_path.unlink()
                if temp_score_path:
                    temp_score_path.unlink()
                if merged_score_path.exists():
                    merged_score_path.unlink()
                print_color(Color.GREEN, "✓ Temporary files cleaned up")
            except Exception as e:
                print_color(Color.YELLOW, f"Warning: Could not clean up temporary files: {e}")
        else:
            print_color(Color.RED, f"❌ Error: Temporary score file was not created for {subfolder_name}")
            print_color(Color.YELLOW, f"  Expected at: {temp_score_path}")
            print_color(Color.YELLOW, f"  This means benchmark failed to process the missing entries")
            return False
    
    # Check if the final score file is complete and evaluate
    print_color(Color.CYAN, "🔍 Validating final score file...")
    validation_result = validate_score_file(score_save_path, protocol_path, verbose=False)
    
    if validation_result.is_valid:
        print_color(Color.GREEN, f"✓ Score file is complete ({validation_result.score_lines}/{validation_result.expected_lines} lines)")
        evaluate_results(score_save_path, protocol_path, summary_file, subfolder_name, eval_config_path, results_folder, benchmark_root, normalized_yaml, comment)
        return True
    elif score_save_path.exists():
        missing_lines = validation_result.expected_lines - validation_result.score_lines
        completion_rate = (validation_result.score_lines / validation_result.expected_lines * 100) if validation_result.expected_lines > 0 else 0
        
        print_color(Color.YELLOW, f"⚠️ Warning: Score file is incomplete for {subfolder_name}")
        print_color(Color.WHITE, f"  Score lines: {validation_result.score_lines}/{validation_result.expected_lines} ({completion_rate:.1f}%)")
        print_color(Color.WHITE, f"  Missing: {missing_lines} lines")
        print_color(Color.WHITE, f"  Protocol subset: {CONSTANTS.protocol_subset}")
        
        # If we just did a resume attempt and still incomplete, it means some files couldn't be processed
        if use_temp_protocol:
            print_color(Color.YELLOW, "  📝 Note: Just attempted resume but still incomplete")
            print_color(Color.YELLOW, "  This likely means some files in the protocol could not be processed:")
            print_color(Color.YELLOW, "    - Corrupted audio files")
            print_color(Color.YELLOW, "    - Missing audio files")
            print_color(Color.YELLOW, "    - Audio format issues")
            print_color(Color.YELLOW, "    - Out of memory errors")
            
            # Check if completion rate is acceptable
            if completion_rate >= CONSTANTS.min_completion_rate:
                print_color(Color.CYAN, f"  ✓ Completion rate ({completion_rate:.1f}%) >= minimum threshold ({CONSTANTS.min_completion_rate}%)")
                print_color(Color.CYAN, f"  💡 Evaluating with {validation_result.score_lines} available samples...")
                try:
                    evaluate_results(score_save_path, protocol_path, summary_file, subfolder_name, eval_config_path, results_folder, benchmark_root, normalized_yaml, comment)
                    print_color(Color.YELLOW, f"  ⚠️ Results for {subfolder_name} (PARTIAL - {completion_rate:.1f}% complete)")
                    return True  # Return True to continue with other datasets
                except Exception as e:
                    print_color(Color.RED, f"  ❌ Could not evaluate partial results: {e}")
            else:
                print_color(Color.RED, f"  ❌ Completion rate ({completion_rate:.1f}%) < minimum threshold ({CONSTANTS.min_completion_rate}%)")
                print_color(Color.YELLOW, f"  To accept partial results, set: export MIN_COMPLETION_RATE={completion_rate:.0f}")
        else:
            print_color(Color.CYAN, "  💡 Tip: Re-run the same command to resume from where it stopped")
        
        return False
    else:
        print_color(Color.RED, f"❌ Error: Score file was not created for {subfolder_name}")
        print_color(Color.YELLOW, f"  Expected at: {score_save_path}")
        return False


def main():
    """Main entry point"""
    # Print banner
    print_banner()
    
    # Parse arguments
    args = parse_arguments()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Print configuration
    print_color(Color.CYAN, f"IS_RANDOM_START: {args.random_start}")
    print_color(Color.CYAN, f"BATCH_SIZE: {args.batch_size}")
    if args.extra_overrides:
        print_color(Color.CYAN, f"EXTRA_OVERRIDES: {' '.join(args.extra_overrides)}")
    eval_config_path = resolve_eval_config_path(args.benchmark_folder, args.eval_config)
    print_color(Color.CYAN, f"EVAL_CONFIG: {eval_config_path if eval_config_path else 'None'}")
    
    # Initialize results directory and summary file
    results_folder, summary_file, normalized_yaml = initialize_results(
        args.results_folder,
        args.comment,
        args.config,
        args.model_path,
        args.adapter_paths,
        args.is_ln,
        args.trim_length,
        eval_config_path,
    )
    
    # Get list of subdirectories
    print_color(Color.CYAN, f"Checking subdirectories in '{args.benchmark_folder}'...")
    subdirs = get_subdirectories(args.benchmark_folder)
    
    total_subfolders = len(subdirs)
    
    if total_subfolders == 0:
        print_color(Color.RED, f"Error: No subdirectories found in '{args.benchmark_folder}'.")
        print_color(Color.YELLOW, "Directory contents:")
        try:
            for item in args.benchmark_folder.iterdir():
                print_color(Color.WHITE, f"  {item}")
        except Exception:
            pass
        sys.exit(1)
    
    print_color(Color.CYAN, f"✓ Starting benchmark with device {args.gpu} and config {args.config}")
    print_color(Color.CYAN, f"✓ Results will be saved to {results_folder}")
    print()
    
    # Process each subfolder in the benchmark folder
    for idx, subfolder in enumerate(subdirs, start=1):
        subfolder_name = subfolder.name
        
        # Display progress
        print_color(Color.YELLOW, "┌─────────────────────────────────────────────────────────────────┐")
        print_color(Color.YELLOW, f"│ Processing dataset: {subfolder_name}")
        print_color(Color.YELLOW, "└─────────────────────────────────────────────────────────────────┘")
        display_progress(idx, total_subfolders, CONSTANTS.progress_bar_width)
        
        # Process the dataset
        success = process_dataset(
            subfolder,
            args.gpu,
            args.config,
            args.model_path,
            results_folder,
            normalized_yaml,
            args.comment,
            args.adapter_paths,
            args.is_ln,
            args.random_start,
            args.trim_length,
            args.batch_size,
            summary_file,
            args.extra_overrides,
            eval_config_path,
            args.benchmark_folder,
        )
        
        if success:
            print_color(Color.GREEN, f"✓ Finished processing {subfolder_name}")
        else:
            print_color(Color.RED, f"❌ Failed processing {subfolder_name}")
        
        print()
    
    # Calculate pooled EER from all datasets
    print_color(Color.MAGENTA, "┌─────────────────────────────────────────────────────────────────┐")
    print_color(Color.MAGENTA, "│                    CALCULATING POOLED EER                       │")
    print_color(Color.MAGENTA, "└─────────────────────────────────────────────────────────────────┘")
    
    calculate_pooled_eer(results_folder, normalized_yaml, args.comment, summary_file, subdirs, eval_config_path)
    
    # Create merged protocol file for reuse
    create_merged_protocol(
        results_folder,
        normalized_yaml,
        args.comment,
        args.config,
        args.model_path,
        summary_file,
        subdirs
    )
    
    # Final summary
    print_color(Color.MAGENTA, "┌─────────────────────────────────────────────────────────────────┐")
    print_color(Color.MAGENTA, "│                       BENCHMARK COMPLETE                        │")
    print_color(Color.MAGENTA, "└─────────────────────────────────────────────────────────────────┘")
    print_color(Color.GREEN, "✓ All benchmarks completed successfully!")
    print_color(Color.CYAN, f"✓ Summary available at: {summary_file}")
    print_color(Color.CYAN, f"✓ Detailed text summary available at: {summary_file.with_name(f'{summary_file.stem}_detailed.txt')}")
    print_color(Color.CYAN, f"✓ Detailed JSONL summary available at: {summary_file.with_name(f'{summary_file.stem}_details.jsonl')}")
    print_color(Color.CYAN, f"✓ Merged protocol available at: {results_folder}/merged_protocol_{normalized_yaml}_{args.comment}.txt")
    print_color(Color.CYAN, f"✓ Merged scores available at: {results_folder}/merged_scores_{normalized_yaml}_{args.comment}.txt")
    print_color(Color.CYAN, f"✓ Protocol metadata available at: {results_folder}/pooled_merged_protocol_{normalized_yaml}_{args.comment}.txt")
    
    # Clean up any remaining temporary files
    cleanup_temp_files(results_folder)
    
    # Pretty print the summary file
    print()
    print_color(Color.YELLOW, "📊 SUMMARY OF RESULTS:")
    print()
    try:
        with open(summary_file, 'r') as f:
            content = f.read().replace('|', '│')
            print_color(Color.WHITE, content)
    except Exception:
        pass
    
    print()
    print_color(Color.GREEN, "Thanks for using the Bulk Benchmark Runner Tool!")


if __name__ == "__main__":
    main()
