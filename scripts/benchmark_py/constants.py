"""
Benchmark Constants Module

Stores configurable constants that can be easily modified via environment variables or code.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BenchmarkConstants:
    """Configuration constants for benchmark operations"""
    
    # Protocol subset configuration
    # Set to the subset name you want to use (e.g., "eval", "test", "dev", etc.)
    # If empty or not found, all lines will be used
    protocol_subset: str = os.getenv("PROTOCOL_SUBSET", "eval")
    
    # Benchmark execution defaults
    default_batch_size: int = int(os.getenv("DEFAULT_BATCH_SIZE", "128"))
    default_trim_length: int = int(os.getenv("DEFAULT_TRIM_LENGTH", "64000"))
    default_is_base_model_path_ln: bool = os.getenv("DEFAULT_IS_BASE_MODEL_PATH_LN", "true").lower() == "true"
    default_is_random_start: bool = os.getenv("DEFAULT_IS_RANDOM_START", "true").lower() == "true"
    
    # Progress bar configuration
    progress_bar_width: int = int(os.getenv("PROGRESS_BAR_WIDTH", "50"))
    
    # File naming patterns
    summary_file_name: str = os.getenv("SUMMARY_FILE_NAME", "summary_results.txt")
    merged_protocol_prefix: str = os.getenv("MERGED_PROTOCOL_PREFIX", "merged_protocol")
    merged_scores_prefix: str = os.getenv("MERGED_SCORES_PREFIX", "merged_scores")
    metadata_file_prefix: str = os.getenv("METADATA_FILE_PREFIX", "pooled_merged_protocol")
    
    # Temporary file patterns
    temp_protocol_prefix: str = os.getenv("TEMP_PROTOCOL_PREFIX", "temp_protocol")
    temp_scores_prefix: str = os.getenv("TEMP_SCORES_PREFIX", "temp_scores")
    temp_protocol_eval_prefix: str = os.getenv("TEMP_PROTOCOL_EVAL_PREFIX", "protocol_eval")
    
    # Minimum completion rate to accept partial results (0-100%)
    # If score file has at least this % of expected lines, accept and evaluate
    min_completion_rate: float = float(os.getenv("MIN_COMPLETION_RATE", "95.0"))
    
    def should_use_protocol_subset(self, protocol_file: str) -> bool:
        """
        Check if protocol subset should be used
        
        Args:
            protocol_file: Path to protocol file
            
        Returns:
            True if subset should be used, False otherwise
        """
        if not self.protocol_subset:
            return False
        
        try:
            with open(protocol_file, 'r') as f:
                for line in f:
                    if self.protocol_subset in line:
                        return True
        except FileNotFoundError:
            return False
        
        return False
    
    def get_protocol_subset_name(self) -> str:
        """Get protocol subset name for display purposes"""
        return self.protocol_subset if self.protocol_subset else "all"


# Global instance
CONSTANTS = BenchmarkConstants()
