"""
Benchmark Execution Module

Handles benchmark command construction and execution.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
import pandas as pd
from .constants import CONSTANTS
from .utils import print_color, Color


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark execution"""
    gpu_number: str
    yaml_config: str
    score_save_path: Path
    data_dir: Path
    protocol_path: Path
    base_model_path: str
    is_base_model_path_ln: bool
    is_random_start: bool
    trim_length: int
    batch_size: int
    adapter_paths: Optional[str] = None
    extra_overrides: Optional[List[str]] = None


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


def construct_benchmark_command(config: BenchmarkConfig) -> list:
    """
    Construct benchmark command as a list for subprocess
    
    Args:
        config: BenchmarkConfig with all parameters
        
    Returns:
        Command as list of strings
    """
    # Convert Path objects to absolute strings
    score_save_path = str(config.score_save_path.absolute())
    data_dir = str(config.data_dir.absolute())
    protocol_path = str(config.protocol_path.absolute())
    
    cmd = [
        'python', 'src/train.py',
        f'experiment={config.yaml_config}',
        f'++model.score_save_path={score_save_path}',
        f'++data.data_dir={data_dir}',
        f'++data.args.protocol_path={protocol_path}',
        '++train=False',
        '++test=True',
        '++model.spec_eval=True',
        f'++data.batch_size={config.batch_size}',
        f'++data.args.random_start={str(config.is_random_start).lower()}',
        f'++data.args.trim_length={config.trim_length}',
        f'++model.base_model_path={config.base_model_path}',
        f'++model.is_base_model_path_ln={str(config.is_base_model_path_ln).lower()}',
    ]
    
    # Add adapter paths if provided
    if config.adapter_paths:
        cmd.append(f'++model.adapter_paths={config.adapter_paths}')

    # Pass through extra Hydra overrides (e.g., +trainer.precision=bf16)
    if config.extra_overrides:
        cmd.extend(config.extra_overrides)

    # Keep Hydra outputs inside the benchmark results folder to avoid permission issues
    has_hydra_run_dir_override = any(
        override.startswith("hydra.run.dir=") or override.startswith("++hydra.run.dir=")
        for override in (config.extra_overrides or [])
    )
    if not has_hydra_run_dir_override:
        hydra_run_dir = config.score_save_path.parent.absolute() / "hydra_runs" / config.score_save_path.stem
        cmd.append(f"hydra.run.dir={hydra_run_dir}")
    
    return cmd


def execute_benchmark(config: BenchmarkConfig) -> bool:
    """
    Execute benchmark command
    
    Args:
        config: BenchmarkConfig with all parameters
        
    Returns:
        True if process completed (regardless of exit code), False if exception
    """
    cmd = construct_benchmark_command(config)
    
    print_color(Color.CYAN, "🔄 Running benchmark...")
    print_color(Color.WHITE, ' '.join(cmd))
    print()
    
    try:
        # Set CUDA_VISIBLE_DEVICES environment variable
        import os
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = config.gpu_number
        env.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
        Path(env['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
        
        # Debug: Show working directory and score path
        print_color(Color.WHITE, f"  Working directory: {os.getcwd()}")
        print_color(Color.WHITE, f"  Score will be saved to: {config.score_save_path}")
        print_color(Color.WHITE, f"  GPU: {env.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        
        # Verify critical files exist before running
        if not config.protocol_path.exists():
            print_color(Color.RED, f"  ❌ ERROR: Protocol file does not exist: {config.protocol_path}")
            return False
        if not config.data_dir.exists():
            print_color(Color.RED, f"  ❌ ERROR: Data directory does not exist: {config.data_dir}")
            return False
        
        print_color(Color.WHITE, f"  ✓ Protocol file exists: {config.protocol_path}")
        print_color(Color.WHITE, f"  ✓ Data directory exists: {config.data_dir}")
        print()
        
        # Flush stdout to ensure all previous output is shown
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Record start time
        import time
        start_time = time.time()
        
        # Build command as string for shell execution (like bash script)
        # This matches the bash script behavior more closely
        cmd_str = ' '.join(cmd)
        
        print_color(Color.CYAN, "  ⏳ Starting benchmark execution...")
        print()
        
        # Execute the command with output streaming to console
        # Use shell=True to match bash script behavior
        result = subprocess.run(
            cmd_str,
            env=env,
            cwd=os.getcwd(),
            shell=True  # Execute via shell like bash script
        )
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Print completion message with exit code and duration for debugging
        if result.returncode == 0:
            print_color(Color.GREEN, f"✓ Benchmark completed successfully (took {duration:.1f}s)")
        else:
            print_color(Color.YELLOW, f"⚠ Benchmark completed with exit code {result.returncode} (took {duration:.1f}s)")
        
        # If it completed too quickly (< 2 seconds), it probably didn't run
        if duration < 2:
            print_color(Color.YELLOW, f"  ⚠️ Warning: Command completed very quickly ({duration:.1f}s)")
            print_color(Color.YELLOW, f"  This might indicate that the model script didn't actually run")
            print_color(Color.YELLOW, f"  Check if there are any import errors or missing dependencies")
        
        # Always return True - let caller check if score file was created
        # This matches bash script behavior
        return True
            
    except KeyboardInterrupt:
        print_color(Color.RED, "\n❌ Benchmark interrupted by user")
        return False
    except Exception as e:
        print_color(Color.RED, f"❌ Error executing benchmark: {e}")
        return False


def evaluate_results(
    score_file: Path,
    protocol_file: Path,
    summary_file: Path,
    dataset_name: str,
    eval_config_path: Optional[Path],
    results_folder: Path,
    benchmark_root: Path,
    normalized_yaml: str,
    comment: str,
) -> bool:
    """
    Evaluate results and extract metrics
    
    Args:
        score_file: Path to score file
        protocol_file: Path to protocol file
        summary_file: Path to summary file
        dataset_name: Name of dataset
        
    Returns:
        True if successful, False otherwise
    """
    print_color(Color.CYAN, "🔄 Evaluating results...")
    
    try:
        result = subprocess.run(
            [
                'python',
                'scripts/score_file_to_eer.py',
                str(score_file),
                str(protocol_file),
                '--output-format',
                'json',
                '--dataset-name',
                dataset_name,
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout.strip()
        
        if not output:
            print_color(Color.RED, f"❌ Error: No output from evaluation script for {dataset_name}")
            return False
        
        payload = json.loads(output)
        detailed_results = payload['results']
        compact_eer = detailed_results['threshold_free']['raw_pooled']['threshold_free'].get('eer')
        compact_auc = detailed_results['threshold_free']['raw_pooled']['threshold_free'].get('roc_auc')
        filled_record = None
        if eval_config_path:
            from benchmark_py.binary_eval import build_eval_frame, evaluate_binary_classification, load_eval_config

            frames = []
            for protocol_path_candidate in sorted(benchmark_root.iterdir()):
                if not protocol_path_candidate.is_dir():
                    continue
                dataset_score_file = results_folder / f"{protocol_path_candidate.name}_{normalized_yaml}_{comment}.txt"
                dataset_protocol_file = protocol_path_candidate / "protocol.txt"
                if not dataset_score_file.exists() or not dataset_protocol_file.exists():
                    continue
                frames.append(
                    build_eval_frame(
                        score_file=dataset_score_file,
                        protocol_file=dataset_protocol_file,
                        dataset_name=protocol_path_candidate.name,
                    )
                )
            if frames:
                all_frame = pd.concat(frames, ignore_index=True)
                config = load_eval_config(eval_config_path)
                filled_results = evaluate_binary_classification(
                    test_frame=all_frame,
                    validation_frame=None,
                    fill_policy=config.get('fill_policy'),
                )
                per_dataset = next(
                    (record for record in filled_results['threshold_free']['per_dataset'] if record['dataset_name'] == dataset_name),
                    None,
                )
                if per_dataset is not None:
                    filled_record = per_dataset
                    compact_eer = per_dataset['threshold_free'].get('eer')
                    compact_auc = per_dataset['threshold_free'].get('roc_auc')
        legacy = payload['legacy_compat']
        compact_accuracy = detailed_results['threshold_based'].get('eer_threshold', {}).get('raw_pooled', {}).get('accuracy')
        if compact_accuracy is None:
            compact_accuracy = legacy.get('accuracy') / 100.0 if legacy.get('accuracy') is not None else float('nan')

        with open(summary_file, 'a') as handle:
            handle.write(
                f"{dataset_name} | "
                f"{_format_percent_from_fraction(compact_eer)} | "
                f"{_format_percent_from_fraction(compact_auc)} | "
                f"{_format_percent_from_fraction(compact_accuracy)}\n"
            )

        detailed_summary_file = summary_file.with_name(f"{summary_file.stem}_detailed.txt")
        with open(detailed_summary_file, 'a') as handle:
            handle.write(f"## Dataset: {dataset_name}\n")
            handle.write(detailed_results['summary_text'])
            handle.write("\n\n")

        detailed_jsonl_file = summary_file.with_name(f"{summary_file.stem}_details.jsonl")
        with open(detailed_jsonl_file, 'a') as handle:
            handle.write(json.dumps({
                'type': 'dataset',
                'dataset_name': dataset_name,
                'score_file': str(score_file),
                'protocol_file': str(protocol_file),
                'payload': payload,
            }))
            handle.write('\n')

        threshold_free = detailed_results['threshold_free']['raw_pooled']['threshold_free']
        eer_threshold_metrics = detailed_results['threshold_based'].get('eer_threshold', {}).get('raw_pooled', {})
        far_1pct_metrics = detailed_results['threshold_based'].get('far_1pct_threshold', {}).get('raw_pooled', {})

        print_color(Color.GREEN, f"✓ Results for {dataset_name}:")
        print_color(Color.WHITE, f"  Legacy EER           : {_format_float(legacy.get('eer'))}")
        print_color(Color.WHITE, f"  Legacy Accuracy      : {_format_float(legacy.get('accuracy'))}")
        print_color(Color.WHITE, f"  Threshold-free EER   : {_format_percent_from_fraction(compact_eer)}")
        print_color(Color.WHITE, f"  ROC-AUC              : {_format_percent_from_fraction(compact_auc)}")
        print_color(Color.WHITE, f"  Accuracy @ val-EER   : {_format_percent_from_fraction(eer_threshold_metrics.get('accuracy'))}")
        print_color(Color.WHITE, f"  FAR @ val-EER        : {_format_percent_from_fraction(eer_threshold_metrics.get('far'))}")
        print_color(Color.WHITE, f"  FRR/MDR @ val-EER    : {_format_percent_from_fraction(eer_threshold_metrics.get('frr'))}")
        print_color(Color.WHITE, f"  MDR @ FAR=1%         : {_format_percent_from_fraction(far_1pct_metrics.get('mdr'))}")
        if filled_record:
            fill_note = filled_record.get('threshold_free', {}).get('note')
            if fill_note:
                print_color(Color.WHITE, f"  Fill note            : {fill_note}")
        print_color(Color.WHITE, f"  Detailed summary     : {detailed_summary_file}")

        return True
            
    except subprocess.CalledProcessError as e:
        print_color(Color.RED, f"❌ Error: Failed to evaluate results for {dataset_name}")
        if e.stderr:
            print_color(Color.YELLOW, f"Details: {e.stderr}")
        return False
    except json.JSONDecodeError as e:
        print_color(Color.RED, f"❌ Error: Invalid JSON from evaluation script for {dataset_name}")
        print_color(Color.YELLOW, f"Details: {e}")
        return False
    except Exception as e:
        print_color(Color.RED, f"❌ Error: {e}")
        return False
