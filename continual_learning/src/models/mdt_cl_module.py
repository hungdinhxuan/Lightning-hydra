import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from avalanche.models import DynamicModule
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from src.models.base.mdt_module import MDTLitModule
from src.models.base.learnable_mdt_module import LearnableMDTLitModule

class MDTCLModule(DynamicModule):
    """Continual Learning version of MDT module with replay capability."""
    
    def __init__(
        self,
        ssl_pretrained_path: str,
        conformer_config: Dict[str, Any],
        replay_buffer_size: int = 1000,
        use_learnable_mdt: bool = True,
        **kwargs
    ):
        """Initialize MDTCL module.
        
        Args:
            ssl_pretrained_path: Path to SSL pretrained model
            conformer_config: Configuration for conformer model
            replay_buffer_size: Size of replay buffer
            use_learnable_mdt: Whether to use learnable MDT or standard MDT
            **kwargs: Additional arguments for MDT module
        """
        super().__init__()
        
        # Initialize base MDT module
        if use_learnable_mdt:
            self.mdt = LearnableMDTLitModule(
                optimizer=None,  # Will be set by training strategy
                scheduler=None,  # Will be set by training strategy
                args=conformer_config,
                ssl_pretrained_path=ssl_pretrained_path,
                **kwargs
            )
        else:
            self.mdt = MDTLitModule(
                optimizer=None,  # Will be set by training strategy
                scheduler=None,  # Will be set by training strategy
                args=conformer_config,
                ssl_pretrained_path=ssl_pretrained_path,
                **kwargs
            )
            
        # Initialize replay buffer
        self.replay_buffer = ReservoirSamplingBuffer(max_size=replay_buffer_size)
        
        # Initialize replay plugin
        self.replay_plugin = ReplayPlugin(
            mem_size=replay_buffer_size,
            batch_size=kwargs.get('batch_size', 32),
            storage_policy=self.replay_buffer
        )
        
        # Task-specific adapters
        self.task_adapters = nn.ModuleDict()
        self.current_task_id = None
        
    def add_task_adapter(self, task_id: int):
        """Add a new task adapter."""
        if str(task_id) not in self.task_adapters:
            self.task_adapters[str(task_id)] = self.mdt.create_adapter()
            
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            task_id: Task identifier for adapter selection
            
        Returns:
            Model output
        """
        # Get base model output
        base_output = self.mdt(x)
        
        # Apply task-specific adapter if task_id is provided
        if task_id is not None:
            task_id = str(task_id)
            if task_id in self.task_adapters:
                return self.task_adapters[task_id](base_output)
            
        return base_output
    
    def adapt(self, experience):
        """Adapt the model to a new experience.
        
        Args:
            experience: The new experience to adapt to
        """
        # Get task ID from experience
        task_id = experience.task_label
        
        # Add task adapter if it doesn't exist
        self.add_task_adapter(task_id)
        
        # Update replay buffer
        self.replay_plugin.update(experience)
        
        # Get replay batch if available
        replay_batch = self.replay_plugin.get_replay_batch()
        
        # Combine new experience with replay batch if available
        if replay_batch is not None:
            # Implement your training logic here
            pass
            
    def get_replay_buffer(self):
        """Get the current replay buffer."""
        return self.replay_buffer
    
    def get_replay_plugin(self):
        """Get the replay plugin."""
        return self.replay_plugin
    
    def get_task_adapters(self):
        """Get all task adapters."""
        return self.task_adapters
    
    def freeze_base_model(self):
        """Freeze the base MDT model parameters."""
        for param in self.mdt.parameters():
            param.requires_grad = False
            
    def unfreeze_base_model(self):
        """Unfreeze the base MDT model parameters."""
        for param in self.mdt.parameters():
            param.requires_grad = True 