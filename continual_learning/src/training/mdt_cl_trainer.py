import torch
from typing import Any, Dict, Optional, List
from avalanche.training import BaseStrategy
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EarlyStoppingPlugin
from avalanche.training.plugins import LRSchedulerPlugin
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from datetime import datetime

class MDTCLTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        max_epochs: int = 100,
        patience: int = 10,
        replay_buffer_size: int = 1000,
        batch_size: int = 32,
        eval_every: int = 1,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        **kwargs
    ):
        """Initialize the MDT Continual Learning Trainer.
        
        Args:
            model: The MDT continual learning model
            optimizer: The optimizer to use
            device: Device to train on
            max_epochs: Maximum number of epochs per experience
            patience: Patience for early stopping
            replay_buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            eval_every: Evaluate every N epochs
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.eval_every = eval_every
        
        # Create directories
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize loggers
        self.loggers = self._setup_loggers()
        
        # Initialize evaluation plugin
        self.eval_plugin = self._setup_eval_plugin()
        
        # Initialize plugins
        self.plugins = self._setup_plugins()
        
        # Initialize strategy
        self.strategy = self._setup_strategy()
        
    def _setup_loggers(self) -> List:
        """Setup loggers for training."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return [
            InteractiveLogger(),
            TextLogger(open(os.path.join(self.log_dir, f"train_{timestamp}.log"), "w")),
            TensorboardLogger(os.path.join(self.log_dir, f"tb_{timestamp}"))
        ]
    
    def _setup_eval_plugin(self) -> EvaluationPlugin:
        """Setup evaluation plugin with metrics."""
        return EvaluationPlugin(
            accuracy_metrics(epoch=True, experience=True, stream=True),
            loss_metrics(epoch=True, experience=True, stream=True),
            loggers=self.loggers
        )
    
    def _setup_plugins(self) -> List:
        """Setup training plugins."""
        plugins = [
            # Early stopping
            EarlyStoppingPlugin(
                patience=self.patience,
                val_stream_name="val_stream",
                metric_name="Top1_Acc_Stream"
            ),
            # Learning rate scheduler
            LRSchedulerPlugin(
                scheduler=ReduceLROnPlateau(
                    self.optimizer,
                    mode='max',
                    patience=self.patience // 2,
                    factor=0.5,
                    verbose=True
                ),
                metric="Top1_Acc_Stream"
            )
        ]
        
        # Add replay plugin if model has replay buffer
        if hasattr(self.model, 'get_replay_plugin'):
            plugins.append(self.model.get_replay_plugin())
            
        return plugins
    
    def _setup_strategy(self) -> BaseStrategy:
        """Setup the training strategy."""
        return BaseStrategy(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.model.criterion,
            train_mb_size=self.batch_size,
            train_epochs=self.max_epochs,
            eval_mb_size=self.batch_size,
            device=self.device,
            plugins=self.plugins,
            evaluator=self.eval_plugin,
            eval_every=self.eval_every
        )
    
    def train(self, train_stream, val_stream=None):
        """Train the model on the given streams.
        
        Args:
            train_stream: Training stream
            val_stream: Validation stream (optional)
        """
        # Train on the stream
        self.strategy.train(train_stream, eval_streams=[val_stream] if val_stream else None)
        
        # Save final checkpoint
        self.save_checkpoint("final")
        
    def save_checkpoint(self, name: str):
        """Save a checkpoint of the model and optimizer.
        
        Args:
            name: Name of the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'strategy_state_dict': self.strategy.state_dict()
        }
        torch.save(
            checkpoint,
            os.path.join(self.checkpoint_dir, f"{name}.pt")
        )
        
    def load_checkpoint(self, name: str):
        """Load a checkpoint of the model and optimizer.
        
        Args:
            name: Name of the checkpoint
        """
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, f"{name}.pt"))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.strategy.load_state_dict(checkpoint['strategy_state_dict'])
        
    def evaluate(self, eval_stream):
        """Evaluate the model on the given stream.
        
        Args:
            eval_stream: Evaluation stream
        """
        return self.strategy.eval(eval_stream)
    
    def get_metrics(self):
        """Get current training metrics."""
        return self.strategy.get_metrics() 