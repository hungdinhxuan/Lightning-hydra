from typing import Any, Dict, Tuple, Union

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy

import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
from src.models.components.xlsr_vib import Model as XLSRVIB
from src.utils.debug import NaNErrorMode
from src.metrics.eer import EERMetric
from torchaudio.models.wav2vec2.utils import import_fairseq_model
from src.models.components.wrapper import W2V2_TA
from src.models.base.adapter_module import AdapterLitModule


class XLSRVIBMultiViewLitModule(AdapterLitModule):
    """XLSR VIB Lightning Module with Multi-View Support.
    
    This module combines the XLSR VIB architecture with multi-view training capabilities,
    where each mini-batch is divided into multiple views (sub-minibatches).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        cp_path: str = None,
        is_train: bool = True,
        score_save_path: str = None,
        **kwargs,
    ) -> None:
        """Initialize XLSRVIBMultiViewLitModule.

        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param args: Arguments for XLSR VIB model configuration.
        :param cp_path: Path to checkpoint for XLSR model.
        :param is_train: Whether the model is in training mode.
        :param score_save_path: Path to save test scores.
        :param kwargs: Additional keyword arguments including weighted_views, adaptive_weights, etc.
        """
        super().__init__(optimizer, scheduler, args, **kwargs)
        
        # Initialize XLSR VIB specific attributes
        self.cp_path = cp_path
        self.is_train = is_train
        self.args = args
        
        # Initialize multi-view specific attributes from kwargs
        weighted_views = kwargs.get("weighted_views", {})
        self.adaptive_weights = kwargs.get("adaptive_weights", False)
        self.score_save_path = score_save_path or kwargs.get("score_save_path", None)
        self.spec_eval = kwargs.get("spec_eval", False)

        # Multi-view accuracy metrics
        self.train_view_acc = {
            view: BinaryAccuracy() for view in weighted_views
        }
        self.val_view_acc = {
            view: BinaryAccuracy() for view in weighted_views
        }
        self.test_view_acc = {}
        
        # For tracking best so far validation accuracy for each view
        self.val_view_acc_best = {
            view: MaxMetric() for view in weighted_views
        }

        # Loss detail tracking for views
        self.train_loss_detail = {}
        self.val_loss_detail = {}
        self.weighted_views = weighted_views
        
        # XLSR VIB specific metrics and attributes
        self.running_loss = 0.0
        self.num_total = 0.0
        
        # Initialize the criteria (loss function)
        self.init_criteria(**kwargs)

    def init_model(self, **kwargs) -> nn.Module:
        """Initialize the XLSR VIB model."""
        return XLSRVIB(self.args, self.cp_path, self.is_train)

    def init_criteria(self, **kwargs) -> torch.nn.Module:
        """Initialize the loss function.
        
        For XLSR VIB, the loss is handled internally by the model's loss method,
        but we also need CrossEntropyLoss for multi-view weighted loss calculation.
        """
        cross_entropy_weight = kwargs.get("cross_entropy_weight", None)
        if cross_entropy_weight is None:
            cross_entropy_weight = torch.tensor([1.0, 1.0])
        else:
            cross_entropy_weight = torch.tensor(cross_entropy_weight)
        self.criterion = torch.nn.CrossEntropyLoss(cross_entropy_weight)
        return self.criterion

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # Reset all metrics
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

        # Reset view-specific metrics
        for k, v in self.val_loss_detail.items():
            self.val_loss_detail[k].reset()

        for k, v in self.val_view_acc.items():
            self.val_view_acc[k].reset()

        for k, v in self.val_view_acc_best.items():
            self.val_view_acc_best[k].reset()

    def model_step(
        self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict, Dict]:
        """Perform a single model step on a batch of multi-view data.

        :param batch: A dictionary where keys are view identifiers and values are tuples
                     containing (input_tensor, target_tensor) for each view.

        :return: A tuple containing (in order):
            - A tensor of total losses across all views.
            - A tensor of concatenated predictions from all views.
            - A tensor of concatenated target labels from all views.
            - A dictionary of loss details for each view.
            - A dictionary of accuracy metrics for each view.
        """
        total_loss = 0.0
        all_preds = []
        all_labels = []
        loss_detail = {}
        view_acc = {}
        
        # Initialize OC_info for XLSR VIB loss calculation
        OC_info = {
            'previous_C': None,
            'previous_num_bona': None
        }

        for view, (info_x_y) in batch.items():
            """
            For multi-view batch:
            - view: view identifier (e.g., 1, 2, 3...)
            - info_x_y: tuple containing (info, batch_x, batch_y)
            """
            info, batch_x, batch_y = info_x_y
            
            # Update batch size calculation
            self.batch_size = batch_x.size(0) * len(self.weighted_views.keys())
            
            # Convert view to string for consistent indexing
            view = str(view)
            
            # Process batch_x similar to original XLSR VIB
            if len(batch_x.shape) == 3:
                batch_x = batch_x.squeeze(0).transpose(0, 1)
            
            batch_y = batch_y.view(-1).type(torch.int64)
            self.num_total += batch_y.shape[0]
            
            # Ensure input tensor is float
            batch_x = batch_x.float()

            # Forward pass through XLSR VIB model
            with NaNErrorMode(
                enabled=False, raise_error=False, print_stats=True, print_nan_index=False
            ):
                batch_out, batch_feat, batch_emb = self.net(batch_x)
                
                # Calculate XLSR VIB losses
                losses = self.net.loss(batch_out, batch_feat,
                                     batch_emb, batch_y, self.args, OC_info)
                
                # Sum up all XLSR VIB losses for this view
                view_loss = 0.0
                for key, value in losses.items():
                    view_loss += value
                    # Track detailed losses
                    loss_key = f"{view}_{key}"
                    self.train_loss_detail[loss_key] = self.train_loss_detail.get(
                        loss_key, 0) + value.item()
                
                # Apply view weighting
                weighted_loss = view_loss * self.weighted_views[view]
                total_loss += weighted_loss
                
                # Get predictions
                _, preds = batch_out.max(dim=1)
                
                # Store view-specific accuracy data
                view_acc[view] = view_acc.get(view, [])
                view_acc[view].append([preds.cpu(), batch_y.cpu()])
                
                # Collect predictions and labels for overall metrics
                all_preds.append(preds)
                all_labels.append(batch_y)
                
                # Store view-specific loss
                loss_detail[view] = loss_detail.get(view, 0) + weighted_loss.item()

        # Update running loss
        self.running_loss += total_loss.item()
        
        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        return total_loss, all_preds, all_labels, loss_detail, view_acc

    def training_step(
        self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of multi-view data."""
        loss, preds, targets, loss_detail, view_acc = self.model_step(batch)

        # Update view-specific accuracy metrics
        for k, v in view_acc.items():
            # Initialize view accuracy metric if needed
            self.train_view_acc[k] = self.train_view_acc.get(k, BinaryAccuracy())
            _preds, _targets = v[0]
            self.train_view_acc[k](_preds, _targets)

        # Update view-specific loss metrics
        for k, v in loss_detail.items():
            # Initialize view loss metric if needed
            self.train_loss_detail[k] = self.train_loss_detail.get(k, MeanMetric())
            self.train_loss_detail[k](v)

        # Update and log overall metrics
        self.train_loss(self.running_loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        # Log view-specific accuracies
        for k, v in self.train_view_acc.items():
            self.log(
                f"train/view_{k}_acc", self.train_view_acc[k].compute(), 
                prog_bar=True, sync_dist=True)

        # Log view-specific losses
        for k, v in self.train_loss_detail.items():
            self.log(
                f"train/view_{k}_loss", self.train_loss_detail[k].compute(), 
                prog_bar=True, sync_dist=True)

        # Log adaptive weights if enabled
        if self.adaptive_weights:
            print(self.weighted_views)
            self.log_dict({f"adaptive_weight_{k}": v for k, v in self.weighted_views.items()}, 
                         on_epoch=True, prog_bar=True, sync_dist=True)

        # Reset XLSR VIB specific metrics
        self.running_loss = 0.0
        self.num_total = 0.0
        self.train_loss_detail = {}

    def validation_step(
        self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of multi-view data."""
        loss, preds, targets, loss_detail, view_acc = self.model_step(batch)

        # Update view-specific accuracy metrics
        for k, v in view_acc.items():
            self.val_view_acc[k] = self.val_view_acc.get(k, BinaryAccuracy())
            _preds, _targets = v[0]
            self.val_view_acc[k](_preds, _targets)

        # Update view-specific loss metrics
        for k, v in loss_detail.items():
            self.val_loss_detail[k] = self.val_loss_detail.get(k, MeanMetric())
            self.val_loss_detail[k](v)

        # Update and log overall metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        # Update best validation accuracy
        acc = self.val_acc.compute()
        self.val_acc_best(acc)

        # Log view-specific losses
        for k, v in self.val_loss_detail.items():
            self.log(
                f"val/view_{k}_loss", self.val_loss_detail[k].compute(), 
                prog_bar=True, sync_dist=True)

        # Log view-specific accuracies and best accuracies
        for k, v in self.val_view_acc.items():
            acc = v.compute()
            self.val_view_acc_best[k](acc)
            self.log(
                f"val/view_{k}_acc", self.val_view_acc[k].compute(), 
                prog_bar=True, sync_dist=True)
            self.log(
                f"val/view_{k}_acc_best", self.val_view_acc_best[k].compute(), 
                prog_bar=True, sync_dist=True)

        self.log("val/acc_best", self.val_acc_best.compute(),
                 sync_dist=True, prog_bar=True, batch_size=self.batch_size)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.
        
        Note: Test step uses the original XLSR VIB format (not multi-view).
        """
        batch_x, utt_id = batch
        batch_out = self.net(batch_x)

        # Write scores to file (same as original XLSR VIB)
        fname_list = list(utt_id)
        score_list = batch_out.data.cpu().numpy().tolist()
            
        with open(self.score_save_path, 'a+') as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write('{} {} {}\n'.format(f, cm[0], cm[1]))

    def on_test_epoch_start(self) -> None:
        """Lightning hook that is called when a test epoch starts."""
        self.net.is_train = False

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization."""
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            print("Using LR Scheduler")
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

 
# if __name__ == "__main__":
#     # Example usage
#     model = XLSRVIBMultiViewLitModule(
#         None, None, None, 
#         cp_path="/data/hungdx/asvspoof5/model/pretrained/xlsr2_300m.pt", 
#         args={
#             "contra_mode": "OC",
#             "loss_type": "CE",
#             "ce_loss_weight": 0,
#             "flag_fix_ssl": False
#         },
#         weighted_views={"1": 1.0, "2": 1.0, "3": 1.0},
#         adaptive_weights=False
#     )
#     print("XLSRVIBMultiViewLitModule created successfully")

