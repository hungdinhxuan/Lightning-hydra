import torch
import numpy as np
from typing import List, Dict, Any, Optional
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer, BalancedExemplarsBuffer
from avalanche.core import SupervisedPlugin

class ReplayStrategy:
    """Base class for replay strategies."""
    def __init__(self, buffer_size: int, **kwargs):
        self.buffer_size = buffer_size
        self.kwargs = kwargs
        
    def get_plugin(self) -> SupervisedPlugin:
        raise NotImplementedError
        
    def get_buffer(self):
        raise NotImplementedError

class ReservoirReplay(ReplayStrategy):
    """Reservoir sampling based replay strategy."""
    def __init__(self, buffer_size: int, **kwargs):
        super().__init__(buffer_size, **kwargs)
        self.buffer = ReservoirSamplingBuffer(max_size=buffer_size)
        
    def get_plugin(self) -> ReplayPlugin:
        return ReplayPlugin(
            mem_size=self.buffer_size,
            batch_size=self.kwargs.get('batch_size', 32),
            storage_policy=self.buffer
        )
        
    def get_buffer(self):
        return self.buffer

class BalancedReplay(ReplayStrategy):
    """Balanced replay strategy that maintains equal number of samples per class."""
    def __init__(self, buffer_size: int, num_classes: int, **kwargs):
        super().__init__(buffer_size, **kwargs)
        self.num_classes = num_classes
        self.buffer = BalancedExemplarsBuffer(
            max_size=buffer_size,
            adaptive_size=True,
            total_num_classes=num_classes
        )
        
    def get_plugin(self) -> ReplayPlugin:
        return ReplayPlugin(
            mem_size=self.buffer_size,
            batch_size=self.kwargs.get('batch_size', 32),
            storage_policy=self.buffer
        )
        
    def get_buffer(self):
        return self.buffer

class GreedyReplay(ReplayStrategy):
    """Greedy replay strategy that selects samples based on importance."""
    def __init__(self, buffer_size: int, **kwargs):
        super().__init__(buffer_size, **kwargs)
        self.buffer = []
        self.importance_scores = []
        
    def update_importance(self, model, batch_x, batch_y):
        """Update importance scores for samples."""
        with torch.no_grad():
            outputs = model(batch_x)
            loss = torch.nn.functional.cross_entropy(outputs, batch_y, reduction='none')
            self.importance_scores.extend(loss.cpu().numpy())
            
    def get_plugin(self) -> ReplayPlugin:
        return ReplayPlugin(
            mem_size=self.buffer_size,
            batch_size=self.kwargs.get('batch_size', 32),
            storage_policy=self
        )
        
    def get_buffer(self):
        return self
        
    def __len__(self):
        return len(self.buffer)
        
    def update(self, experience):
        """Update buffer with new experience."""
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            # Replace least important sample
            min_idx = np.argmin(self.importance_scores)
            self.buffer[min_idx] = experience
            self.importance_scores[min_idx] = 0  # Reset importance score
            
    def get_replay_batch(self):
        """Get batch of samples for replay."""
        if len(self.buffer) == 0:
            return None
            
        indices = np.random.choice(len(self.buffer), 
                                 size=min(len(self.buffer), self.kwargs.get('batch_size', 32)),
                                 replace=False)
        return [self.buffer[i] for i in indices] 