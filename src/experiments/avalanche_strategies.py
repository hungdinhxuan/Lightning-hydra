from typing import Dict, Any, Optional
import torch
from omegaconf import DictConfig
from avalanche.training import Naive, Replay, EWC, LwF
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from avalanche.training.plugins import EvaluationPlugin
from src.models.base.mdt_module import MDTLitModule

class StrategyFactory:
    """Factory class for creating Avalanche training strategies."""
    
    @staticmethod
    def create_strategy(
        strategy_name: str,
        model: MDTLitModule,
        cfg: DictConfig,
        eval_plugin: EvaluationPlugin,
        plugins: list
    ) -> Any:
        """Create and return an Avalanche training strategy.
        
        Args:
            strategy_name: Name of the strategy to create
            model: The MDT model to train
            cfg: Configuration containing strategy parameters
            eval_plugin: Evaluation plugin for metrics
            plugins: List of additional plugins to use
            
        Returns:
            An instance of the requested Avalanche strategy
        """
        strategy_name = strategy_name.lower()
        
        # Extract the actual PyTorch model from Lightning wrapper
        pytorch_model = model.net if hasattr(model, 'net') else model
        
        # Create optimizer for the PyTorch model
        optimizer = torch.optim.Adam(
            pytorch_model.parameters(), 
            lr=cfg.model.optimizer.lr,
            weight_decay=cfg.model.optimizer.get('weight_decay', 0.0001)
        )
        
        # Use the criterion from the Lightning model
        criterion = model.criterion
        
        # Common parameters for all strategies
        common_params = {
            "model": pytorch_model,
            "optimizer": optimizer,
            "criterion": criterion,
            "train_mb_size": cfg.data.batch_size,
            "train_epochs": cfg.trainer.max_epochs,
            "eval_mb_size": cfg.data.batch_size,
            "evaluator": eval_plugin,
            "plugins": plugins
        }
        
        # Strategy-specific parameters
        if strategy_name == "naive":
            return Naive(**common_params)
            
        elif strategy_name == "replay":
            storage_policy = ReservoirSamplingBuffer(
                max_size=cfg.avalanche.replay_buffer_size
            )
            return Replay(
                **common_params,
                storage_policy=storage_policy
            )
            
        elif strategy_name == "ewc":
            return EWC(
                **common_params,
                ewc_lambda=cfg.avalanche.ewc_lambda
            )
            
        elif strategy_name == "lwf":
            return LwF(
                **common_params,
                alpha=cfg.avalanche.lwf_alpha,
                temperature=cfg.avalanche.lwf_temperature
            )
            
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

# Dictionary mapping strategy names to their descriptions
STRATEGY_DESCRIPTIONS = {
    "naive": "Basic continual learning without any special mechanisms",
    "replay": "Uses a replay buffer to store and replay past experiences",
    "ewc": "Elastic Weight Consolidation - prevents catastrophic forgetting by constraining important weights",
    "lwf": "Learning without Forgetting - uses knowledge distillation to maintain performance on old tasks"
}

def get_available_strategies() -> Dict[str, str]:
    """Get a dictionary of available strategies and their descriptions.
    
    Returns:
        Dictionary mapping strategy names to their descriptions
    """
    return STRATEGY_DESCRIPTIONS 