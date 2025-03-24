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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

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
            self.batch_size = x.size(0) * len(self.weighted_views.keys())  # Update batch size 

            view = str(view)  # Convert view to string for indexing

            # Ensure the input tensor is of type float
            x = x.float()

            logits = self.forward(x)
            # print size of logits and size of y
            loss = self.criterion(
                logits, y) * self.weighted_views[str(view)]  # Weighted loss
            train_loss += loss
            preds = torch.argmax(logits, dim=1)

            view_acc[view] = view_acc.get(view, [])

            # Append predictions and target to view_acc
            view_acc[view].append([preds.cpu(), y.cpu()])

            # Collect predictions and labels
            all_preds.append(preds)
            all_labels.append(y)

            # Intialize loss_detail dictionary
            loss_detail[view] = loss_detail.get(view, 0) + loss.item()

        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        return train_loss, all_preds, all_labels, loss_detail, view_acc

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

        # Log current adaptive_weights
        if self.adaptive_weights:
            print(self.weighted_views)
            self.log_dict({f"adaptive_weight_{k}": v for k, v in self.weighted_views.items(
            )}, on_epoch=True, prog_bar=True, sync_dist=True)

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
