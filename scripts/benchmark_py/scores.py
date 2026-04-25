"""
Benchmark Scores Module

Handles score file merging operations.
"""

from pathlib import Path
from typing import List, Tuple, Dict
from .utils import print_color, Color


def parse_score_line(line: str) -> tuple:
    """
    Parse a score line into components
    
    Handles paths with spaces and quotes:
    - Format 1: <path> <score1> <score2>
    - Format 2: <path> <score>
    - Format 3: "<path with spaces>" <score1> <score2>
    
    Args:
        line: Line from score file
        
    Returns:
        Tuple of (filename, score1, score2, original_line) or None if invalid
    """
    original_line = line
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    
    # Check if path is quoted
    if line.startswith('"') or line.startswith("'"):
        quote_char = line[0]
        close_quote_idx = line.find(quote_char, 1)
        if close_quote_idx == -1:
            return None
        
        filename = line[1:close_quote_idx]
        remainder = line[close_quote_idx + 1:].strip()
        parts = remainder.split()
        
        try:
            if len(parts) >= 2:
                score1 = float(parts[0])
                score2 = float(parts[1])
                return (filename, score1, score2, original_line.strip())
            elif len(parts) == 1:
                score = float(parts[0])
                return (filename, score, score, original_line.strip())
        except ValueError:
            return None
        
        return None
    
    # Parse from right: last 2 numbers are scores, rest is path
    parts = line.split()
    if len(parts) >= 3:
        try:
            # Try to parse last 2 as scores
            score1 = float(parts[-2])
            score2 = float(parts[-1])
            filename = ' '.join(parts[:-2])
            return (filename, score1, score2, original_line.strip())
        except ValueError:
            return None
    elif len(parts) >= 2:
        try:
            # Try to parse last 1 as score
            score = float(parts[-1])
            filename = ' '.join(parts[:-1])
            return (filename, score, score, original_line.strip())
        except ValueError:
            return None
    
    return None


def read_score_file(score_file: Path) -> Dict[str, Tuple[float, float, str]]:
    """
    Read scores from file into dictionary
    
    Args:
        score_file: Path to score file
        
    Returns:
        Dictionary mapping filename to (score1, score2, original_line)
    """
    scores = {}
    
    if not score_file.exists():
        return scores
    
    try:
        with open(score_file, 'r') as f:
            for line_text in f:
                parsed = parse_score_line(line_text)
                if parsed:
                    filename, score1, score2, original_line = parsed
                    scores[filename] = (score1, score2, original_line)
    except Exception as e:
        print_color(Color.RED, f"Error reading score file {score_file}: {e}")
    
    return scores


def merge_score_files(
    original_score: Path,
    new_score: Path,
    merged_score: Path
) -> bool:
    """
    Merge two score files
    
    Args:
        original_score: Path to original score file
        new_score: Path to new score file
        merged_score: Path to output merged score file
        
    Returns:
        True if successful, False otherwise
    """
    print_color(Color.CYAN, "🔄 Merging score files...")
    
    try:
        # Create backup of original score file
        if original_score.exists():
            backup_path = original_score.with_suffix('.txt.backup')
            with open(original_score, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
        
        # Read both score files
        original_scores = read_score_file(original_score)
        new_scores = read_score_file(new_score)
        
        # Combine and deduplicate (keep new scores if duplicate)
        score_dict = {}
        
        # Add original scores
        for filename, (score1, score2, line) in original_scores.items():
            score_dict[filename] = (score1, score2, line)
        
        # Add/overwrite with new scores
        for filename, (score1, score2, line) in new_scores.items():
            score_dict[filename] = (score1, score2, line)
        
        # Sort by filename and write
        with open(merged_score, 'w') as f:
            for filename in sorted(score_dict.keys()):
                _, _, line = score_dict[filename]
                f.write(line + '\n')
        
        # Replace original with merged
        if merged_score.exists():
            with open(merged_score, 'r') as src, open(original_score, 'w') as dst:
                dst.write(src.read())
            print_color(Color.GREEN, "✓ Score files merged successfully")
            return True
        
        return False
        
    except Exception as e:
        print_color(Color.RED, f"Error merging score files: {e}")
        return False


def read_scores(score_path: Path) -> Dict[str, Tuple[float, float]]:
    """
    Read scores from score file into dictionary
    
    Args:
        score_path: Path to score file
        
    Returns:
        Dictionary mapping file_id to (bonafide_score, spoof_score)
    """
    scores = {}
    
    if not score_path.exists():
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
        print_color(Color.RED, f"Error reading score file {score_path}: {e}")
    
    return scores
