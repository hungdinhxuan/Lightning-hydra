"""
Benchmark Execution Module

Handles benchmark command construction and execution.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
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
    dataset_name: str
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
        # Call the evaluation script
        result = subprocess.run(
            ['python', 'scripts/score_file_to_eer.py', str(score_file), str(protocol_file)],
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout.strip()
        
        if not output:
            print_color(Color.RED, f"❌ Error: No output from evaluation script for {dataset_name}")
            return False
        
        # Parse the output
        parts = output.split()
        if len(parts) >= 5:
            min_score = parts[0]
            max_score = parts[1]
            threshold = parts[2]
            eer = parts[3]
            accuracy = parts[4]
            
            # Format output for summary file
            with open(summary_file, 'a') as f:
                f.write(f"{dataset_name} | {eer} | {min_score} | {max_score} | {threshold} | {accuracy}\n")
            
            # Display results
            print_color(Color.GREEN, f"✓ Results for {dataset_name}:")
            print_color(Color.WHITE, f"  EER      : {eer}")
            print_color(Color.WHITE, f"  Accuracy : {accuracy}")
            print_color(Color.WHITE, f"  Threshold: {threshold}")
            print_color(Color.WHITE, f"  Min Score: {min_score}")
            print_color(Color.WHITE, f"  Max Score: {max_score}")
            
            return True
        else:
            print_color(Color.RED, f"❌ Error: Invalid output format from evaluation script")
            return False
            
    except subprocess.CalledProcessError as e:
        print_color(Color.RED, f"❌ Error: Failed to evaluate results for {dataset_name}")
        if e.stderr:
            print_color(Color.YELLOW, f"Details: {e.stderr}")
        return False
    except Exception as e:
        print_color(Color.RED, f"❌ Error: {e}")
        return False
