"""
Benchmark Protocol Module

Handles protocol file operations including missing entry detection.
"""

import tempfile
from pathlib import Path
from typing import Optional, Tuple
from .constants import CONSTANTS
from .utils import print_color, Color, count_non_empty_lines


def parse_protocol_line(line: str) -> Tuple[str, str, str]:
    """
    Parse a single protocol line of the form:
        <relative_path> <subset> <label>
    
    Handles paths with spaces by:
    1. Checking for quoted paths (starts with " or ')
    2. Using rsplit(maxsplit=2) to split from right
    
    Examples:
        - "path/file.wav eval bonafide" -> ("path/file.wav", "eval", "bonafide")
        - "path with spaces/file.wav eval spoof" -> ("path with spaces/file.wav", "eval", "spoof")
        - '"path with spaces/file.wav" eval spoof' -> ("path with spaces/file.wav", "eval", "spoof")
    
    Returns:
        Tuple of (rel_path, subset, label) or None if invalid
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    
    # Check if path is quoted (starts with " or ')
    if line.startswith('"') or line.startswith("'"):
        quote_char = line[0]
        # Find closing quote
        close_quote_idx = line.find(quote_char, 1)
        if close_quote_idx == -1:
            # Malformed quoted path
            return None
        
        rel_path = line[1:close_quote_idx]
        remainder = line[close_quote_idx + 1:].strip()
        parts = remainder.split()
        
        if len(parts) != 2:
            return None
        
        subset, label = parts
        return rel_path, subset, label
    
    # Parse from right to left: last 2 whitespaces separate subset and label
    # This allows paths to contain spaces without quoting
    parts = line.rsplit(maxsplit=2)
    if len(parts) != 3:
        return None
    
    rel_path, subset, label = parts
    return rel_path, subset, label


def create_missing_protocol(
    score_file: Path,
    protocol_file: Path,
    temp_protocol: Path
) -> int:
    """
    Create temporary protocol file with missing entries (optimized for sequential evaluation)
    
    Args:
        score_file: Path to existing score file
        protocol_file: Path to protocol file
        temp_protocol: Path to temporary protocol file to create
        
    Returns:
        Number of missing entries (0 if all complete, negative if error)
    """
    print_color(Color.CYAN, "🔍 Analyzing missing entries (optimized for sequential evaluation)...")
    
    if not protocol_file.exists():
        print_color(Color.RED, f"  Error: Protocol file not found at {protocol_file}")
        return -1
    
    # Count existing score lines (exclude comments and empty lines)
    # Match bash behavior: grep -c "^[^[:space:]]*[[:space:]]"
    # This means: line must have at least one non-whitespace followed by whitespace (i.e., at least 2 fields)
    existing_lines = 0
    if score_file.exists():
        try:
            with open(score_file, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith('#'):
                        continue
                    # Check if line has at least one whitespace (i.e., at least 2 fields)
                    if ' ' in stripped or '\t' in stripped:
                        existing_lines += 1
        except Exception as e:
            print_color(Color.RED, f"  Error reading score file: {e}")
            return -1
    
    print_color(Color.WHITE, f"  Existing score lines: {existing_lines}")
    print_color(Color.WHITE, f"  Protocol subset: {CONSTANTS.protocol_subset if CONSTANTS.protocol_subset else 'all'}")
    
    # Create temporary file for protocol subset
    with tempfile.NamedTemporaryFile(mode='w', prefix='protocol_eval_', suffix='.txt', delete=False) as temp_f:
        temp_protocol_subset = Path(temp_f.name)
    
    try:
        # Extract subset entries from protocol file
        subset_line_count = 0
        if CONSTANTS.should_use_protocol_subset(str(protocol_file)):
            # If protocol has the specified subset, use only subset lines
            with open(protocol_file, 'r') as src, open(temp_protocol_subset, 'w') as dst:
                for line in src:
                    parsed = parse_protocol_line(line)
                    if parsed:
                        _, subset, _ = parsed
                        if subset == CONSTANTS.protocol_subset:
                            dst.write(line)
                            subset_line_count += 1
            print_color(Color.WHITE, f"  Using protocol subset: {CONSTANTS.protocol_subset}")
        else:
            # If no subset found or subset is empty, use all lines
            with open(protocol_file, 'r') as src, open(temp_protocol_subset, 'w') as dst:
                for line in src:
                    if parse_protocol_line(line) is not None:
                        dst.write(line)
                        subset_line_count += 1
            print_color(Color.WHITE, 
                       f"  Using all protocol lines (subset '{CONSTANTS.get_protocol_subset_name()}' not found or not specified)")
        
        print_color(Color.WHITE, f"  Total {CONSTANTS.get_protocol_subset_name()} lines in protocol: {subset_line_count}")
        
        # Calculate missing lines (sequential evaluation - just skip processed lines)
        missing_count = subset_line_count - existing_lines
        
        if missing_count <= 0:
            print_color(Color.GREEN, "  No missing entries found.")
            temp_protocol.touch()  # Create empty temp protocol
            return 0
        
        # Create temporary protocol file with remaining entries (starting from existing_lines + 1)
        with open(temp_protocol_subset, 'r') as src:
            lines = src.readlines()
            # Skip the first existing_lines entries
            if existing_lines >= len(lines):
                print_color(Color.YELLOW, f"  Warning: existing_lines ({existing_lines}) >= total lines ({len(lines)})")
                temp_protocol.touch()
                return 0
            
            remaining_lines = lines[existing_lines:]
            
            with open(temp_protocol, 'w') as dst:
                dst.writelines(remaining_lines)
        
        print_color(Color.YELLOW, 
                   f"  Found {missing_count} missing entries (lines {existing_lines + 1} to {subset_line_count})")
        
        return missing_count
        
    except Exception as e:
        print_color(Color.RED, f"  Error creating missing protocol: {e}")
        return -1
    finally:
        # Clean up temporary files
        try:
            temp_protocol_subset.unlink()
        except Exception:
            pass


def extract_eval_subset(protocol_file: Path, output_file: Path) -> None:
    """
    Extract protocol subset from protocol file
    
    Args:
        protocol_file: Path to input protocol file
        output_file: Path to output file for subset
    """
    if CONSTANTS.should_use_protocol_subset(str(protocol_file)):
        # If protocol has the specified subset, use only subset lines
        with open(protocol_file, 'r') as src, open(output_file, 'w') as dst:
            for line in src:
                parsed = parse_protocol_line(line)
                if parsed:
                    _, subset, _ = parsed
                    if subset == CONSTANTS.protocol_subset:
                        dst.write(line)
    else:
        # If no subset found or subset is empty, use all lines
        with open(protocol_file, 'r') as src, open(output_file, 'w') as dst:
            for line in src:
                if parse_protocol_line(line) is not None:
                    dst.write(line)


def read_protocol_eval_subset(protocol_path: Path) -> list:
    """
    Read evaluation subset from protocol file
    
    Args:
        protocol_path: Path to protocol file
        
    Returns:
        List of tuples (file_id, label, subset)
    """
    protocol_entries = []
    
    if not protocol_path.exists():
        return protocol_entries
    
    try:
        # First pass: check if there are any lines with the specified subset
        has_subset = False
        with open(protocol_path, 'r') as f:
            for line in f:
                parsed = parse_protocol_line(line)
                if parsed and CONSTANTS.protocol_subset:
                    _, subset, _ = parsed
                    if subset == CONSTANTS.protocol_subset:
                        has_subset = True
                        break
        
        # Second pass: read appropriate lines
        with open(protocol_path, 'r') as f:
            for line in f:
                parsed = parse_protocol_line(line)
                if not parsed:
                    continue
                
                file_id, subset, label = parsed
                
                # If there's a subset filter, only process those lines
                # If no subset filter, process all lines
                if not CONSTANTS.protocol_subset or not has_subset or subset == CONSTANTS.protocol_subset:
                    protocol_entries.append((file_id, label, subset))
                        
    except Exception as e:
        print_color(Color.RED, f"Error reading protocol file {protocol_path}: {e}")
    
    return protocol_entries
