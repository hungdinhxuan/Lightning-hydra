import torch
import torch.nn as nn
from typing import Any, Dict, Tuple, Union
from avalanche.models import DynamicModule
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy
from src.models.components.xlsr_conformertcm_baseline import Model as XLSRConformerTCM

class XLSRConformertcmMDTCL(DynamicModule):
    def __init__(
        self,
        ssl_pretrained_path: str,
        conformer_config: dict,
        replay_buffer_size: int = 1000,
        weighted_views: Dict[str, float] = None,
        adaptive_weights: bool = False,
        **kwargs
    ):
        """Initialize the continual learning version of XLSR-ConformerTCM with multi-duration training.
        
        Args:
            ssl_pretrained_path (str): Path to the SSL pretrained model
            conformer_config (dict): Configuration for the conformer model
            replay_buffer_size (int): Size of the replay buffer
            weighted_views (Dict[str, float]): Dictionary mapping view names to their weights
            adaptive_weights (bool): Whether to use adaptive weights for different durations
        """
        super().__init__()
        
        # Initialize the base model
        self.model = XLSRConformerTCM(conformer_config, ssl_pretrained_path)
        
        # Initialize replay buffer
        self.replay_buffer = ReservoirSamplingBuffer(max_size=replay_buffer_size)
        
        # Initialize replay plugin
        self.replay_plugin = ReplayPlugin(
            mem_size=replay_buffer_size,
            batch_size=kwargs.get('batch_size', 32),
            storage_policy=self.replay_buffer
        )
        
        # Multi-duration training settings
        self.weighted_views = weighted_views or {"1": 1.0}  # Default to single view
        self.adaptive_weights = adaptive_weights
        
        # Initialize metrics
        self.train_view_acc = {
            view: BinaryAccuracy() for view in self.weighted_views
        }
        self.val_view_acc = {
            view: BinaryAccuracy() for view in self.weighted_views
        }
        self.val_view_acc_best = {
            view: MaxMetric() for view in self.weighted_views
        }
        
        self.train_loss_detail = {}
        self.val_loss_detail = {}
        
        # Initialize loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Freeze SSL layers
        self._freeze_ssl_layers()
        
    def _freeze_ssl_layers(self):
        """Freeze the SSL layers to prevent catastrophic forgetting."""
        for param in self.model.ssl_model.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)
    
    def model_step(
        self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, list]]:
        """Perform a single model step on a batch of data with multiple durations.
        
        Args:
            batch: Dictionary mapping view names to (input, target) tuples
            
        Returns:
            Tuple containing:
            - Total loss
            - All predictions
            - All targets
            - Loss details per view
            - Accuracy metrics per view
        """
        train_loss = 0.0
        all_preds = []
        all_labels = []
        loss_detail = {}
        view_acc = {}
        
        for view, (x, y) in batch.items():
            view = str(view)  # Convert view to string for indexing
            
            # Forward pass
            logits = self.forward(x)
            
            # Calculate weighted loss
            loss = self.criterion(logits, y) * self.weighted_views[view]
            train_loss += loss
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Store view-specific metrics
            view_acc[view] = view_acc.get(view, [])
            view_acc[view].append([preds.cpu(), y.cpu()])
            
            # Collect predictions and labels
            all_preds.append(preds)
            all_labels.append(y)
            
            # Store loss details
            loss_detail[view] = loss_detail.get(view, 0) + loss.item()
        
        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        
        return train_loss, all_preds, all_labels, loss_detail, view_acc
    
    def adapt(self, experience):
        """Adapt the model to a new experience.
        
        Args:
            experience: The new experience to adapt to
        """
        # Update replay buffer with new experience
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
    
    def update_metrics(self, loss, preds, targets, loss_detail, view_acc, phase="train"):
        """Update metrics for the current phase.
        
        Args:
            loss: Total loss
            preds: All predictions
            targets: All targets
            loss_detail: Loss details per view
            view_acc: Accuracy metrics per view
            phase: Current phase ("train" or "val")
        """
        # Update view-specific accuracy
        for view, acc_data in view_acc.items():
            metric = self.train_view_acc if phase == "train" else self.val_view_acc
            _preds, _targets = acc_data[0]
            metric[view](_preds, _targets)
        
        # Update loss details
        for view, loss_val in loss_detail.items():
            metric = self.train_loss_detail if phase == "train" else self.val_loss_detail
            metric[view] = metric.get(view, MeanMetric())
            metric[view](loss_val)
    
    def get_metrics(self, phase="train"):
        """Get current metrics for the specified phase.
        
        Args:
            phase: Current phase ("train" or "val")
            
        Returns:
            Dictionary of current metrics
        """
        metrics = {}
        
        # Get view-specific accuracy
        acc_metrics = self.train_view_acc if phase == "train" else self.val_view_acc
        for view, metric in acc_metrics.items():
            metrics[f"{phase}/view_{view}_acc"] = metric.compute()
        
        # Get loss details
        loss_metrics = self.train_loss_detail if phase == "train" else self.val_loss_detail
        for view, metric in loss_metrics.items():
            metrics[f"{phase}/view_{view}_loss"] = metric.compute()
        
        return metrics 