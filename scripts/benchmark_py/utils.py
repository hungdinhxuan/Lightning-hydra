"""
Benchmark Utilities Module

Provides colored output, progress tracking, and UI functions.
"""

import sys
import time
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List
from enum import Enum


class Color(Enum):
    """Terminal color codes"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[0;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


def print_color(color: Color, text: str) -> None:
    """
    Print colored text to stdout
    
    Args:
        color: Color enum value
        text: Text to print
    """
    print(f"{color.value}{text}{Color.RESET.value}")


def display_progress(current: int, total: int, width: int = 50) -> None:
    """
    Display a progress bar
    
    Args:
        current: Current progress count
        total: Total count
        width: Width of progress bar (default: 50)
    """
    if total == 0:
        print_color(Color.RED, "Error: No subdirectories found to process.")
        return
    
    percentage = (current * 100) // total
    completed = (width * current) // total
    remaining = width - completed
    
    bar = f"{Color.WHITE.value}["
    bar += f"{Color.GREEN.value}{'=' * completed}"
    
    if completed < width:
        bar += ">"
        bar += " " * (remaining - 1)
    
    bar += f"{Color.WHITE.value}] {percentage}% ({current}/{total}){Color.RESET.value}"
    print(bar)


def print_banner() -> None:
    """Print the application banner"""
    # Clear screen
    print("\033[H\033[J", end='')
    
    print_color(Color.MAGENTA, "┌─────────────────────────────────────────────────────────────────┐")
    print_color(Color.MAGENTA, "│               🚀 BULK BENCHMARK RUNNER TOOL 🚀                  │")
    print_color(Color.MAGENTA, "└─────────────────────────────────────────────────────────────────┘")
    print()


def print_usage() -> str:
    """Return usage information string"""
    usage = """
┌─────────────────────────────────────────────────────────────────┐
│                 Bulk Benchmark Runner Script                    │
└─────────────────────────────────────────────────────────────────┘

Usage: python benchmark.py [OPTIONS]

Required Parameters:
  -g, --gpu GPU               GPU identifier (index like 0/1 or MIG UUID)
  -c, --config CONFIG         YAML config file path (e.g., cnsl/xlsr_vib_large_corpus)
  -b, --benchmark-folder DIR  Bulk benchmark folder path
  -m, --model-path PATH       Base model path
  -r, --results-folder DIR    Results folder path
  -n, --comment COMMENT       Comment to note

Optional Parameters:
  -a, --adapter-paths PATH    Adapter paths (optional)
  -l, --is-ln BOOL           Whether to use Lightning checkpoint loading (default: true)
  -s, --random-start BOOL    Whether to use random start (default: true)
  -t, --trim-length INT      Trim length for data processing (default: 64000)
"""
    return usage


class SpinnerContext:
    """Context manager for showing a spinner during long operations"""
    
    def __init__(self, description: str = "Processing"):
        self.description = description
        self.spinner_chars = ['-', '\\', '|', '/']
        self.running = False
        
    def __enter__(self):
        self.running = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if exc_type is None:
            print(f"\r{Color.GREEN.value}✓ {self.description} completed{' ' * 20}{Color.RESET.value}")
        return False
    
    def spin(self) -> None:
        """Show a single frame of the spinner"""
        if self.running:
            for char in self.spinner_chars:
                if not self.running:
                    break
                print(f"\r{Color.CYAN.value}⏳ {self.description}: {char}{Color.RESET.value}", end='', flush=True)
                time.sleep(0.1)


def cleanup_temp_files(results_folder: Optional[Path] = None) -> None:
    """
    Clean up temporary files
    
    Args:
        results_folder: Optional results folder path to clean
    """
    print_color(Color.CYAN, "🧹 Cleaning up temporary files...")
    
    patterns = [
        "temp_protocol_*.txt",
        "temp_scores_*.txt",
        "protocol_eval_*.txt",
        "protocol_ids_*.txt",
        "existing_ids_*.txt",
        "missing_ids_*.txt",
        "existing_scores_*.txt",
    ]
    
    # Clean up in results folder if specified
    if results_folder:
        for pattern in patterns:
            for file in results_folder.glob(pattern):
                try:
                    file.unlink()
                except Exception:
                    pass
    
    # Clean up in /tmp
    tmp_dir = Path("/tmp")
    for pattern in patterns:
        for file in tmp_dir.glob(pattern):
            try:
                file.unlink()
            except Exception:
                pass


def count_non_empty_lines(file_path: Path) -> int:
    """
    Count non-empty lines in a file
    
    Args:
        file_path: Path to file
        
    Returns:
        Number of non-empty lines
    """
    if not file_path.exists():
        return 0
    
    count = 0
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        return 0
    
    return count


def normalize_yaml_name(yaml_config: str) -> str:
    """
    Normalize YAML config name for use in file names
    
    Args:
        yaml_config: YAML config path
        
    Returns:
        Normalized name with slashes replaced by underscores
    """
    return yaml_config.replace('/', '_')
