from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch

@dataclass
class TrainingConfig:
    # Data parameters
    data_dir: str
    protocol_path: str
    batch_size: int = 32
    num_workers: int = 4
    
    # Model parameters
    ssl_pretrained_path: str
    conformer_config: Dict[str, Any]
    
    # Training parameters
    learning_rate: float = 1e-4
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Replay parameters
    replay_strategy: str = "reservoir"  # Options: "reservoir", "balanced", "greedy"
    replay_buffer_size: int = 1000
    replay_batch_size: int = 32
    
    # Logging parameters
    log_dir: str = "experiments/logs"
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "experiments/logs/tensorboard"
    
    # Evaluation parameters
    eval_every: int = 1  # Evaluate every N experiences
    save_every: int = 1  # Save checkpoint every N experiences
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "protocol_path": self.protocol_path,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "ssl_pretrained_path": self.ssl_pretrained_path,
            "conformer_config": self.conformer_config,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "device": self.device,
            "replay_strategy": self.replay_strategy,
            "replay_buffer_size": self.replay_buffer_size,
            "replay_batch_size": self.replay_batch_size,
            "log_dir": self.log_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "tensorboard_dir": self.tensorboard_dir,
            "eval_every": self.eval_every,
            "save_every": self.save_every
        } 