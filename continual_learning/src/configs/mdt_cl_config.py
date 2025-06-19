from dataclasses import dataclass
from typing import Dict, Any, Optional
from omegaconf import MISSING

@dataclass
class ModelConfig:
    ssl_pretrained_path: str = MISSING
    conformer_config: Dict[str, Any] = MISSING
    replay_buffer_size: int = 1000
    weighted_views: Dict[str, float] = None
    adaptive_weights: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-5
    eval_every: int = 1
    num_workers: int = 4
    device: str = "cuda"

@dataclass
class DataConfig:
    data_dir: str = MISSING
    protocol_dir: str = MISSING
    cache_dir: Optional[str] = None
    enable_cache: bool = False
    wav_samp_rate: int = 16000
    trim_length: int = 66800
    padding_type: str = "repeat"
    random_start: bool = False
    views: int = 3
    view_padding_configs: Dict[str, Any] = None

@dataclass
class LoggingConfig:
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    save_top_k: int = 3
    monitor: str = "val/acc"
    mode: str = "max"

@dataclass
class MDTCLConfig:
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig() 