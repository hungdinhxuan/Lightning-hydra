from typing import Any, Dict, Tuple
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy
from typing import Union
import torch
from src.utils import load_ln_model_weights
from peft import LoraConfig, TaskType
import peft
from peft import PeftModel
from src.models.base.adapter_module import AdapterLitModule
from torch import nn
import torch.nn.functional as F


class MDTLitModule(AdapterLitModule):
  
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
     
        super().__init__(optimizer, scheduler, args, **kwargs)
        self.init_criteria(**kwargs)
        
        self.score_save_path = kwargs.get("score_save_path", None)
        self.spec_eval = kwargs.get("spec_eval", False)
        
        # Initialize learnable view weights
        self.init_view_weights(**kwargs)
    
    def init_view_weights(self, **kwargs):
        """Initialize learnable view weights as parameters."""
        weighted_views = kwargs.get("weighted_views", {})
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
        self.learnable_weights = kwargs.get("learnable_weights", False)
        self.weight_normalization = kwargs.get("weight_normalization", "softmax")
        self.weight_temperature = kwargs.get("weight_temperature", 1.0)
        
        if self.learnable_weights:
            view_keys = list(weighted_views.keys())
            initial_weights = torch.tensor([weighted_views[key] for key in view_keys], dtype=torch.float32)
            
            # Initialize raw weights based on normalization type
            if self.weight_normalization == "softmax":
                # Initialize with log probabilities for softmax
                # Add small random noise to break symmetry
                log_weights = torch.log(initial_weights + 1e-8) + torch.randn_like(initial_weights) * 0.1
                self.raw_view_weights = nn.Parameter(log_weights, requires_grad=True)
            elif self.weight_normalization == "sigmoid":
                # Initialize with logits for sigmoid
                self.raw_view_weights = nn.Parameter(torch.logit(initial_weights + 1e-8), requires_grad=True)
            else:
                # Direct parameterization with small random noise
                self.raw_view_weights = nn.Parameter(initial_weights + torch.randn_like(initial_weights) * 0.1, requires_grad=True)
            
            self.view_keys = view_keys
            
            # Scale learning rate for weights (typically 10x higher than model learning rate)
            self.weight_lr_scale = kwargs.get("weight_lr_scale", 10.0)
        else:
            self.static_weights = weighted_views
            
    def get_current_weights(self) -> Dict[str, float]:
        """Get current view weights (normalized if learnable)."""
        if not self.learnable_weights:
            return self.static_weights
        
        if self.weight_normalization == "softmax":
            normalized_weights = F.softmax(self.raw_view_weights / self.weight_temperature, dim=0)
        elif self.weight_normalization == "sigmoid":
            normalized_weights = torch.sigmoid(self.raw_view_weights)
        else:
            normalized_weights = F.relu(self.raw_view_weights) + 1e-8
        
        # Normalize to sum to 1
        normalized_weights = normalized_weights / normalized_weights.sum()
        
        return {key: weight.item() for key, weight in zip(self.view_keys, normalized_weights)}
    
    def log_view_weights(self, stage: str = "train"):
        """Log current view weights for monitoring."""
        if self.learnable_weights:
            current_weights = self.get_current_weights()
            for view, weight in current_weights.items():
                self.log(f"{stage}_weight_view_{view}", weight, prog_bar=True, sync_dist=True)
                 
    def init_criteria(self, **kwargs) -> torch.nn.Module:
        """
            Initialize the loss function with the given arguments. This method is used to initialize the loss
            function with the given arguments. The loss function is initialized with the given arguments and the
            loss function is returned.
            
            Base model doesn't implement this method. This method should be implemented in the derived
            model class.
        """
        cross_entropy_weight = kwargs.get("cross_entropy_weight", None)
        if cross_entropy_weight is None:
            cross_entropy_weight = torch.tensor(cross_entropy_weight)
        else:
            cross_entropy_weight = torch.tensor([1.0, 1.0])
        self.criterion = torch.nn.CrossEntropyLoss(cross_entropy_weight)
        
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
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
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
            - A dictionary of loss details for each view.
            - A dictionary of accuracy metrics for each view.
        """
        train_loss = 0.0
        all_preds = []
        all_labels = []
        loss_detail = {}
        view_acc = {}
    
        current_weights = self.get_current_weights()
        
        for view, (x, y) in batch.items():
            '''
            In the case of multi-view, the batch is a dictionary with the view as the key and the value
            is a tuple of the input tensor and target tensor.


                view: int
                x: sub-minibatch of input tensor
                y: sub-minibatch of target tensor

                Example:
                 view: 1
                 x: shape (batch_size, view * sample_rate)
                 y: shape (batch_size, 1)
            '''
            self.batch_size = x.size(0) * len(current_weights.keys())  # Update batch size 

            view = str(view)  # Convert view to string for indexing

            # Ensure the input tensor is of type float
            x = x.float()

            logits = self.forward(x)
            
            # Get the weight for this view
            if self.learnable_weights:
                # Get the normalized weight tensor directly
                if self.weight_normalization == "softmax":
                    weights = F.softmax(self.raw_view_weights / self.weight_temperature, dim=0)
                elif self.weight_normalization == "sigmoid":
                    weights = torch.sigmoid(self.raw_view_weights)
                else:
                    weights = F.relu(self.raw_view_weights) + 1e-8
                weights = weights / weights.sum()
                view_idx = self.view_keys.index(view)
                weight = weights[view_idx]
            else:
                weight = current_weights[view]

            # Compute loss with weight directly in the computation
            loss = self.criterion(logits, y) * weight
            train_loss += loss
            preds = torch.argmax(logits, dim=1)

            view_acc[view] = view_acc.get(view, [])

            # Append predictions and target to view_acc
            view_acc[view].append([preds.cpu(), y.cpu()])

            # Collect predictions and labels
            all_preds.append(preds)
            all_labels.append(y)

            # Initialize loss_detail dictionary
            loss_detail[view] = loss_detail.get(view, 0) + loss.item()

        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        return train_loss, all_preds, all_labels, loss_detail, view_acc

    def on_train_epoch_start(self) -> None:
        """Lightning hook that is called when a training epoch starts."""
        # Only clear cache if memory usage is high
        if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.9 * torch.cuda.get_device_properties(0).total_memory:
            torch.cuda.empty_cache()
            
    def on_validation_epoch_start(self) -> None:
        """Lightning hook that is called when a validation epoch starts."""
        # Only clear cache if memory usage is high
        if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.9 * torch.cuda.get_device_properties(0).total_memory:
            torch.cuda.empty_cache()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, loss_detail, view_acc = self.model_step(batch)

        # Update train_view_acc and log metrics
        for k, v in view_acc.items():
            # Initialize train_view_acc dictionary
            self.train_view_acc[k] = self.train_view_acc.get(
                k, BinaryAccuracy())
            _preds, _targets = v[0]
            self.train_view_acc[k](_preds, _targets)

        # Update train_loss_detail and log metrics
        for k, v in loss_detail.items():
            # Initialize train_loss_detail dictionary
            self.train_loss_detail[k] = self.train_loss_detail.get(
                k, MeanMetric())
            self.train_loss_detail[k](v)

        # Update and log train loss and accuracy
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        # Log weight information and memory usage every 100 batches
        if self.learnable_weights and batch_idx % 100 == 0:
            current_weights = self.get_current_weights()
            for view, weight in current_weights.items():
                self.log(f"train/weight_view_{view}", weight, on_step=True, on_epoch=False, 
                        prog_bar=False, sync_dist=True)
            if self.raw_view_weights.grad is not None:
                for view, grad in zip(self.view_keys, self.raw_view_weights.grad):
                    self.log(f"train/weight_grad_view_{view}", grad.item(), on_step=True, on_epoch=False,
                            prog_bar=False, sync_dist=True)
            
            # Log memory usage only when it's high
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**2
                memory_reserved = torch.cuda.memory_reserved() / 1024**2
                if memory_allocated > 0.8 * torch.cuda.get_device_properties(0).total_memory / 1024**2:
                    self.log("train/memory_allocated", memory_allocated, 
                            on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
                    self.log("train/memory_reserved", memory_reserved,
                            on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        # Return loss for backpropagation
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        for k, v in self.train_view_acc.items():
            self.log(
                f"train/view_{k}_acc", self.train_view_acc[k].compute(), prog_bar=True, sync_dist=True)

        for k, v in self.train_loss_detail.items():
            self.log(
                f"train/view_{k}_loss", self.train_loss_detail[k].compute(), prog_bar=True, sync_dist=True)
        
        # Log view weights
        self.log_view_weights(stage="train")
            
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, loss_detail, view_acc = self.model_step(batch)

        # Update val_view_acc and log metrics
        for k, v in view_acc.items():
            # Initialize val_view_acc dictionary
            self.val_view_acc[k] = self.val_view_acc.get(k, BinaryAccuracy())
            _preds, _targets = v[0]
            self.val_view_acc[k](_preds, _targets)
            
        # Update val_loss_detail and log metrics
        for k, v in loss_detail.items():
            # Initialize val_loss_detail dictionary
            self.val_loss_detail[k] = self.val_loss_detail.get(k, MeanMetric())
            self.val_loss_detail[k](v)

        # Update and log val loss and accuracy
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        # Log weight information every 100 batches
        if self.learnable_weights and batch_idx % 100 == 0:
            current_weights = self.get_current_weights()
            for view, weight in current_weights.items():
                self.log(f"val/weight_view_{view}", weight, on_step=True, on_epoch=False, 
                        prog_bar=False, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc

        for k, v in self.val_loss_detail.items():
            self.log(
                f"val/view_{k}_loss", self.val_loss_detail[k].compute(), prog_bar=True, sync_dist=True)

        for k, v in self.val_view_acc.items():
            acc = v.compute()
            self.val_view_acc_best[k](acc)
            self.log(
                f"val/view_{k}_acc", self.val_view_acc[k].compute(), prog_bar=True, sync_dist=True)
            self.log(
                f"val/view_{k}_acc_best", self.val_view_acc_best[k].compute(), prog_bar=True, sync_dist=True)

        self.log("val/acc_best", self.val_acc_best.compute(),
                 sync_dist=True, prog_bar=True, batch_size=self.batch_size)

        # Log final weights at the end of validation
        if self.learnable_weights:
            current_weights = self.get_current_weights()
            for view, weight in current_weights.items():
                self.log(f"val/final_weight_view_{view}", weight, on_epoch=True, 
                        prog_bar=True, sync_dist=True)
        
        # Clear any cached memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers with different learning rates for weights and model.
        
        This method extends the base class's optimizer configuration to handle learnable weights
        with a different learning rate. If learnable_weights is False, it falls back to the
        base class's configuration.
        
        Returns:
            Dict[str, Any]: A dictionary containing the configured optimizer and scheduler
        """
        if not self.learnable_weights:
            return super().configure_optimizers()
            
        # Get base optimizer configuration
        base_config = super().configure_optimizers()
        base_optimizer = base_config["optimizer"]
        base_lr = base_optimizer.param_groups[0]["lr"]
        
        # Create parameter groups with different learning rates
        weight_params = {
            'params': self.raw_view_weights,
            'lr': base_lr * self.weight_lr_scale
        }
        model_params = {
            'params': [p for n, p in self.named_parameters() if 'raw_view_weights' not in n],
            'lr': base_lr
        }
        
        # Create new optimizer with parameter groups
        optimizer = type(base_optimizer)(
            [weight_params, model_params],
            **{k: v for k, v in base_optimizer.param_groups[0].items() if k != 'params'}
        )
        
        # Handle scheduler if it exists
        if "lr_scheduler" in base_config:
            scheduler_config = base_config["lr_scheduler"]
            scheduler = type(scheduler_config["scheduler"])(
                optimizer,
                **{k: v for k, v in scheduler_config["scheduler"].__dict__.items() 
                   if k not in ['optimizer', 'base_lrs']}
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": scheduler_config.get("monitor", "val/loss"),
                    "interval": scheduler_config.get("interval", "epoch"),
                    "frequency": scheduler_config.get("frequency", 1),
                },
            }
            
        return {"optimizer": optimizer}

    