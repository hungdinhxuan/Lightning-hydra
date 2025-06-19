import torch
import torch.nn as nn
import torch.optim as optim
from avalanche.training.supervised import Naive
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.training.checkpoint import maybe_load_checkpoint, save_checkpoint
import os
from typing import Optional, Dict, Any
from src.models.xlsr_conformertcm_cl import XLSRConformerTCMCL
from src.data.avalanche_datamodule import AvalancheDataModule
from src.training.replay_strategy import ReplayStrategy, ReservoirReplay, BalancedReplay, GreedyReplay
from src.configs.training_config import TrainingConfig

class ContinualLearningTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize data module
        self.data_module = AvalancheDataModule(
            data_dir=config.data_dir,
            protocol_path=config.protocol_path,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        self.data_module.prepare_data()
        self.benchmark = self.data_module.get_benchmark()
        
        # Initialize model
        self.model = XLSRConformerTCMCL(
            ssl_pretrained_path=config.ssl_pretrained_path,
            conformer_config=config.conformer_config,
            replay_buffer_size=config.replay_buffer_size
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize replay strategy
        self.replay_strategy = self._init_replay_strategy()
        
        # Initialize evaluation plugin
        self.eval_plugin = EvaluationPlugin(
            accuracy_metrics(epoch=True, experience=True, stream=True),
            loss_metrics(epoch=True, experience=True, stream=True),
            loggers=[
                InteractiveLogger(),
                TensorboardLogger(config.tensorboard_dir)
            ]
        )
        
        # Initialize training strategy
        self.strategy = Naive(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            train_mb_size=config.batch_size,
            train_epochs=config.epochs,
            eval_mb_size=config.batch_size,
            device=self.device,
            plugins=[self.replay_strategy.get_plugin()],
            evaluator=self.eval_plugin
        )
        
    def _init_replay_strategy(self) -> ReplayStrategy:
        """Initialize the replay strategy based on config."""
        if self.config.replay_strategy == "reservoir":
            return ReservoirReplay(
                buffer_size=self.config.replay_buffer_size,
                batch_size=self.config.replay_batch_size
            )
        elif self.config.replay_strategy == "balanced":
            return BalancedReplay(
                buffer_size=self.config.replay_buffer_size,
                num_classes=2,  # Binary classification for spoofing detection
                batch_size=self.config.replay_batch_size
            )
        elif self.config.replay_strategy == "greedy":
            return GreedyReplay(
                buffer_size=self.config.replay_buffer_size,
                batch_size=self.config.replay_batch_size
            )
        else:
            raise ValueError(f"Unknown replay strategy: {self.config.replay_strategy}")
            
    def train(self):
        """Main training loop."""
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # Training loop
        for experience in self.benchmark.train_stream:
            print(f"Training on experience {experience.current_experience}")
            
            # Train
            self.strategy.train(experience)
            
            # Evaluate if needed
            if experience.current_experience % self.config.eval_every == 0:
                self.strategy.eval(self.benchmark.test_stream)
            
            # Save checkpoint if needed
            if experience.current_experience % self.config.save_every == 0:
                save_checkpoint(
                    self.strategy,
                    os.path.join(
                        self.config.checkpoint_dir,
                        f"checkpoint_exp_{experience.current_experience}.pth"
                    )
                )
                
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint."""
        maybe_load_checkpoint(self.strategy, checkpoint_path)
        
    def evaluate(self):
        """Evaluate the model on the test stream."""
        return self.strategy.eval(self.benchmark.test_stream) 