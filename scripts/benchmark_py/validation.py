"""
Benchmark Validation Module

Handles score file validation and completeness checking.
"""

from pathlib import Path
from typing import Tuple
from .constants import CONSTANTS
from .utils import print_color, Color, count_non_empty_lines
from .protocol import parse_protocol_line


class ValidationResult:
    """Result of validation with status and details"""
    
    def __init__(self, is_valid: bool, score_lines: int, expected_lines: int, message: str = ""):
        self.is_valid = is_valid
        self.score_lines = score_lines
        self.expected_lines = expected_lines
        self.message = message
    
    def __bool__(self):
        return self.is_valid


def validate_score_file(score_file: Path, protocol_file: Path, verbose: bool = True) -> ValidationResult:
    """
    Validate score file completeness
    
    Args:
        score_file: Path to score file
        protocol_file: Path to protocol file
        verbose: Whether to print validation details (default: True)
        
    Returns:
        ValidationResult with validation status and details
    """
    if not score_file.exists():
        return ValidationResult(False, 0, 0, "Score file doesn't exist")
    
    if not protocol_file.exists():
        return ValidationResult(False, 0, 0, "Protocol file doesn't exist")
    
    # Count lines in score file (excluding empty lines and comments)
    # Match bash behavior: grep -c "^[^[:space:]]*[[:space:]]"
    # This means: line must have at least one non-whitespace followed by whitespace (i.e., at least 2 fields)
    score_lines = 0
    try:
        with open(score_file, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                # Check if line has at least one whitespace (i.e., at least 2 fields)
                if ' ' in stripped or '\t' in stripped:
                    score_lines += 1
    except Exception as e:
        if verbose:
            print_color(Color.RED, f"  Error reading score file: {e}")
        return ValidationResult(False, 0, 0, f"Error reading score file: {e}")
    
    # Count subset lines in protocol file
    subset_lines = 0
    protocol_subset_used = False
    
    try:
        if CONSTANTS.should_use_protocol_subset(str(protocol_file)):
            # Count lines containing the subset using proper parsing
            protocol_subset_used = True
            with open(protocol_file, 'r') as f:
                for line in f:
                    parsed = parse_protocol_line(line)
                    if parsed:
                        _, subset, _ = parsed
                        if subset == CONSTANTS.protocol_subset:
                            subset_lines += 1
        
        # If no subset found or subset is empty, count all lines (fallback)
        if subset_lines == 0:
            protocol_subset_used = False
            with open(protocol_file, 'r') as f:
                for line in f:
                    if parse_protocol_line(line) is not None:
                        subset_lines += 1
    except Exception as e:
        if verbose:
            print_color(Color.RED, f"  Error reading protocol file: {e}")
        return ValidationResult(False, score_lines, 0, f"Error reading protocol file: {e}")
    
    if verbose:
        print_color(Color.WHITE, f"  Score file lines: {score_lines}")
        print_color(Color.WHITE, f"  Expected lines ({CONSTANTS.get_protocol_subset_name()} subset): {subset_lines}")
        if protocol_subset_used:
            print_color(Color.WHITE, f"  Using protocol subset: {CONSTANTS.protocol_subset}")
        else:
            print_color(Color.WHITE, f"  Using all protocol lines (no subset filtering)")
    
    if score_lines == subset_lines and score_lines > 0:
        return ValidationResult(True, score_lines, subset_lines, "Valid and complete")
    else:
        return ValidationResult(False, score_lines, subset_lines, "Incomplete or corrupted")


def check_protocol_exists(protocol_file: Path) -> bool:
    """
    Check if protocol file exists
    
    Args:
        protocol_file: Path to protocol file
        
    Returns:
        True if exists, False otherwise
    """
    exists = protocol_file.exists()
    if not exists:
        print_color(Color.RED, f"⚠️ Warning: Protocol file not found at {protocol_file}")
    return exists
