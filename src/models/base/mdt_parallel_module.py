from typing import Any, Dict, Tuple, Union, List

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy

from src.utils import load_ln_model_weights
from peft import LoraConfig, TaskType
import peft
from peft import PeftModel
from src.models.base.adapter_module import AdapterLitModule


class MDTParallelLitModule(AdapterLitModule):
    """
    A parallel version of MDTLitModule that processes multiple views simultaneously.
    
    Instead of iterating through views sequentially, this module:
    1. Concatenates all view inputs into a single batch
    2. Runs ONE forward pass on the combined batch
    3. Splits outputs back by view for loss calculation
    
    This significantly improves performance by:
    - Reducing the number of forward passes from N (views) to 1
    - Better GPU utilization through larger batch sizes
    - Reduced Python loop overhead
    """
  
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
     
        super().__init__(optimizer, scheduler, args, **kwargs)
        self.init_criteria(**kwargs)
        
        weighted_views = kwargs.get("weighted_views", {})
        self.adaptive_weights = kwargs.get("adaptive_weights", False)
        self.score_save_path = kwargs.get("score_save_path", None)
        self.spec_eval = kwargs.get("spec_eval", False)

        self.train_view_acc = {
            view: BinaryAccuracy() for view in weighted_views
        }
        self.val_view_acc = {
            view: BinaryAccuracy() for view in weighted_views
        }
        self.test_view_acc = {}
        # for tracking best so far validation accuracy for each view
        self.val_view_acc_best = {
            view: MaxMetric() for view in weighted_views
        }

        self.train_loss_detail = {}
        self.val_loss_detail = {}
        self.weighted_views = weighted_views
        
        
    def init_criteria(self, **kwargs) -> torch.nn.Module:
        """
        Initialize the loss function with the given arguments.
        
        Args:
            **kwargs: Keyword arguments including cross_entropy_weight
        """
        cross_entropy_weight = kwargs.get("cross_entropy_weight", None)
        if cross_entropy_weight is None:
            cross_entropy_weight = torch.tensor([1.0, 1.0])
        else:
            cross_entropy_weight = torch.tensor(cross_entropy_weight)
        self.criterion = torch.nn.CrossEntropyLoss(weight=cross_entropy_weight, reduction='none')
        
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

        for k, v in self.val_loss_detail.items():
            self.val_loss_detail[k].reset()

        for k, v in self.val_view_acc.items():
            self.val_view_acc[k].reset()

        for k, v in self.val_view_acc_best.items():
            self.val_view_acc_best[k].reset()

    def model_step(
        self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict, Dict]:
        """
        Perform a PARALLEL model step on a batch of data with multiple views.
        
        This method concatenates all view inputs, runs a single forward pass,
        and then splits the outputs for per-view loss calculation.

        Args:
            batch: A dictionary where keys are view names and values are 
                   tuples of (input_tensor, target_tensor).

        Returns:
            A tuple containing (in order):
                - Total weighted loss (tensor)
                - Concatenated predictions (tensor)
                - Concatenated target labels (tensor)
                - Loss details per view (dict)
                - Accuracy metrics per view (dict)
        """
        # Collect all inputs and labels, tracking view boundaries
        all_inputs = []
        all_labels = []
        view_sizes = {}  # Track the size of each view's batch
        view_order = []  # Maintain order of views
        
        for view, (x, y) in batch.items():
            view = str(view)
            view_order.append(view)
            view_sizes[view] = x.size(0)
            
            # Ensure input tensor is float
            x = x.float()
            
            all_inputs.append(x)
            all_labels.append(y)
        
        # Concatenate all inputs and labels into single tensors
        combined_inputs = torch.cat(all_inputs, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)
        
        # Update batch size for logging
        self.batch_size = combined_inputs.size(0)
        
        # Single forward pass for all views
        combined_logits = self.forward(combined_inputs)
        
        # Compute per-sample losses (without reduction)
        per_sample_losses = self.criterion(combined_logits, combined_labels)
        
        # Get predictions
        all_preds = torch.argmax(combined_logits, dim=1)
        
        # Split results back by view and compute weighted losses
        total_loss = torch.tensor(0.0, device=combined_inputs.device)
        loss_detail = {}
        view_acc = {}
        
        start_idx = 0
        for view in view_order:
            size = view_sizes[view]
            end_idx = start_idx + size
            
            # Extract this view's losses, predictions, and labels
            view_losses = per_sample_losses[start_idx:end_idx]
            view_preds = all_preds[start_idx:end_idx]
            view_labels = combined_labels[start_idx:end_idx]
            
            # Compute mean loss for this view and apply weight
            view_loss = view_losses.mean() * self.weighted_views[view]
            total_loss = total_loss + view_loss
            
            # Store loss detail
            loss_detail[view] = view_loss.item()
            
            # Store accuracy data for this view
            view_acc[view] = [[view_preds.cpu(), view_labels.cpu()]]
            
            start_idx = end_idx

        return total_loss, all_preds, combined_labels, loss_detail, view_acc

    def model_step_chunked(
        self, 
        batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        max_chunk_size: int = 1024
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict, Dict]:
        """
        Memory-efficient parallel model step that processes data in chunks.
        
        Use this method when dealing with very large batches that might
        cause out-of-memory errors.

        Args:
            batch: A dictionary of view data
            max_chunk_size: Maximum number of samples to process at once

        Returns:
            Same as model_step()
        """
        # Collect all inputs and labels
        all_inputs = []
        all_labels = []
        view_sizes = {}
        view_order = []
        
        for view, (x, y) in batch.items():
            view = str(view)
            view_order.append(view)
            view_sizes[view] = x.size(0)
            all_inputs.append(x.float())
            all_labels.append(y)
        
        combined_inputs = torch.cat(all_inputs, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)
        
        self.batch_size = combined_inputs.size(0)
        total_samples = combined_inputs.size(0)
        
        # Process in chunks to manage memory
        all_logits = []
        for i in range(0, total_samples, max_chunk_size):
            chunk_inputs = combined_inputs[i:i + max_chunk_size]
            chunk_logits = self.forward(chunk_inputs)
            all_logits.append(chunk_logits)
        
        combined_logits = torch.cat(all_logits, dim=0)
        
        # Rest is same as model_step
        per_sample_losses = self.criterion(combined_logits, combined_labels)
        all_preds = torch.argmax(combined_logits, dim=1)
        
        total_loss = torch.tensor(0.0, device=combined_inputs.device)
        loss_detail = {}
        view_acc = {}
        
        start_idx = 0
        for view in view_order:
            size = view_sizes[view]
            end_idx = start_idx + size
            
            view_losses = per_sample_losses[start_idx:end_idx]
            view_preds = all_preds[start_idx:end_idx]
            view_labels = combined_labels[start_idx:end_idx]
            
            view_loss = view_losses.mean() * self.weighted_views[view]
            total_loss = total_loss + view_loss
            
            loss_detail[view] = view_loss.item()
            view_acc[view] = [[view_preds.cpu(), view_labels.cpu()]]
            
            start_idx = end_idx

        return total_loss, all_preds, combined_labels, loss_detail, view_acc

    def training_step(
        self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single training step on a batch of data.

        Args:
            batch: A batch of data (dictionary of view -> (inputs, targets))
            batch_idx: The index of the current batch.
            
        Returns:
            A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, loss_detail, view_acc = self.model_step(batch)

        # Update train_view_acc and log metrics
        for k, v in view_acc.items():
            self.train_view_acc[k] = self.train_view_acc.get(k, BinaryAccuracy())
            _preds, _targets = v[0]
            self.train_view_acc[k](_preds, _targets)

        # Update train_loss_detail and log metrics
        for k, v in loss_detail.items():
            self.train_loss_detail[k] = self.train_loss_detail.get(k, MeanMetric())
            self.train_loss_detail[k](v)

        # Update and log train loss and accuracy
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        for k, v in self.train_view_acc.items():
            self.log(
                f"train/view_{k}_acc", self.train_view_acc[k].compute(), 
                prog_bar=True, sync_dist=True
            )

        for k, v in self.train_loss_detail.items():
            self.log(
                f"train/view_{k}_loss", self.train_loss_detail[k].compute(), 
                prog_bar=True, sync_dist=True
            )

        # Log current adaptive_weights
        if self.adaptive_weights:
            print(self.weighted_views)
            self.log_dict(
                {f"adaptive_weight_{k}": v for k, v in self.weighted_views.items()}, 
                on_epoch=True, prog_bar=True, sync_dist=True
            )

    def validation_step(
        self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> None:
        """
        Perform a single validation step on a batch of data.

        Args:
            batch: A batch of data (dictionary of view -> (inputs, targets))
            batch_idx: The index of the current batch.
        """
        loss, preds, targets, loss_detail, view_acc = self.model_step(batch)

        # Update val_view_acc and log metrics
        for k, v in view_acc.items():
            self.val_view_acc[k] = self.val_view_acc.get(k, BinaryAccuracy())
            _preds, _targets = v[0]
            self.val_view_acc[k](_preds, _targets)
            
        # Update val_loss_detail and log metrics
        for k, v in loss_detail.items():
            self.val_loss_detail[k] = self.val_loss_detail.get(k, MeanMetric())
            self.val_loss_detail[k](v)

        # Update and log val loss and accuracy
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        acc = self.val_acc.compute()
        self.val_acc_best(acc)

        for k, v in self.val_loss_detail.items():
            self.log(
                f"val/view_{k}_loss", self.val_loss_detail[k].compute(), 
                prog_bar=True, sync_dist=True
            )

        for k, v in self.val_view_acc.items():
            acc = v.compute()
            self.val_view_acc_best[k](acc)
            self.log(
                f"val/view_{k}_acc", self.val_view_acc[k].compute(), 
                prog_bar=True, sync_dist=True
            )
            self.log(
                f"val/view_{k}_acc_best", self.val_view_acc_best[k].compute(), 
                prog_bar=True, sync_dist=True
            )

        self.log("val/acc_best", self.val_acc_best.compute(),
                 sync_dist=True, prog_bar=True, batch_size=self.batch_size)
