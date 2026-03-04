"""
Configuration file for parallel audio processing
"""

import multiprocessing as mp
from pathlib import Path


class ParallelConfig:
    """Configuration class for parallel audio processing"""
    
    # Default paths
    DEFAULT_ROOT_DIR = "/nvme2/hungdx/Lightning-hydra/data/SSBoll_resample"
    DEFAULT_SOURCE_DIR = "/nvme2/hungdx/Lightning-hydra/data/wildspoof_challenge_benchmark/record"
    
    # Processing settings
    DEFAULT_NAME_PROCESSING = "ssBoll_py_parallel"
    DEFAULT_POST_NAME = "_ssBoll_py"
    
    # Parallel processing settings
    DEFAULT_NUM_PROCESSES = None  # Will use CPU count
    MAX_PROCESSES = mp.cpu_count()
    MIN_PROCESSES = 1
    
    # Audio processing settings
    SAMPLE_RATE = 16000
    IS_DURATION = 0.25  # Initial silence duration for SSBoll79
    
    # Categories to process
    DEFAULT_CATEGORIES = [
        "benign/en",
        "spoof/bark/en", 
        "spoof/vits/en",
        "spoof/xtts_v1.1/en",
        "spoof/xtts_v2/en"
    ]
    
    # Progress reporting
    ENABLE_PROGRESS_BAR = True
    PROGRESS_UPDATE_INTERVAL = 10  # Update every N files
    
    # Error handling
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    
    # Memory management
    CHUNK_SIZE = 100  # Process files in chunks to manage memory
    
    @classmethod
    def get_optimal_processes(cls, num_files=None):
        """Get optimal number of processes based on system and workload"""
        cpu_count = mp.cpu_count()
        
        if num_files is None:
            return cpu_count
        
        # Don't use more processes than files
        optimal = min(cpu_count, num_files)
        
        # For very small workloads, use fewer processes
        if num_files < 4:
            optimal = min(2, num_files)
        
        return max(cls.MIN_PROCESSES, optimal)
    
    @classmethod
    def validate_paths(cls, root_dir, source_dir):
        """Validate that paths exist and are accessible"""
        root_path = Path(root_dir)
        source_path = Path(source_dir)
        
        if not root_path.exists():
            raise FileNotFoundError(f"Root directory does not exist: {root_dir}")
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
        
        if not root_path.is_dir():
            raise NotADirectoryError(f"Root path is not a directory: {root_dir}")
        
        if not source_path.is_dir():
            raise NotADirectoryError(f"Source path is not a directory: {source_dir}")
        
        return True
    
    @classmethod
    def get_memory_usage_info(cls):
        """Get information about available memory for processing"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                'total': memory.total,
                'available': memory.available,
                'percent_used': memory.percent,
                'recommended_max_processes': max(1, int(memory.available / (1024**3) * 2))  # 2 processes per GB
            }
        except ImportError:
            return {
                'total': 'Unknown',
                'available': 'Unknown', 
                'percent_used': 'Unknown',
                'recommended_max_processes': cls.MAX_PROCESSES
            }


# Example usage configurations
CONFIGS = {
    'fast': {
        'num_processes': mp.cpu_count(),
        'chunk_size': 50,
        'enable_progress_bar': True
    },
    'memory_efficient': {
        'num_processes': max(1, mp.cpu_count() // 2),
        'chunk_size': 25,
        'enable_progress_bar': True
    },
    'debug': {
        'num_processes': 1,
        'chunk_size': 10,
        'enable_progress_bar': True
    }
}


def get_config(config_name='fast'):
    """Get a specific configuration"""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    
    config = CONFIGS[config_name].copy()
    return config


def print_system_info():
    """Print system information for parallel processing"""
    print("System Information:")
    print(f"CPU cores: {mp.cpu_count()}")
    
    memory_info = ParallelConfig.get_memory_usage_info()
    print(f"Memory: {memory_info['total'] / (1024**3):.1f} GB total, "
          f"{memory_info['available'] / (1024**3):.1f} GB available")
    print(f"Recommended max processes: {memory_info['recommended_max_processes']}")
    
    print(f"Default processes: {ParallelConfig.DEFAULT_NUM_PROCESSES or mp.cpu_count()}")
    print(f"Available configurations: {list(CONFIGS.keys())}")


if __name__ == "__main__":
    print_system_info()
