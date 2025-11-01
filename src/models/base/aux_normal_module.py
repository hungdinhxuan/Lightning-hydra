from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy, MulticlassAccuracy

from typing import Union

import torch
from src.models.base.adapter_module import AdapterLitModule

class AuxNormalLitModule(AdapterLitModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
        """Initialize a `AuxNormalLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(optimizer, scheduler, args, **kwargs)
        # Initialize metrics
        self.init_criteria(**kwargs)
        
        # Get auxiliary training parameters
        self.scale_aux = kwargs.get("scale_aux", 0.1)
        aux_num_classes = kwargs.get("aux_num_classes", 11)
        
        # Initialize auxiliary metrics
        if aux_num_classes > 2:
            self.train_aux_acc = MulticlassAccuracy(num_classes=aux_num_classes)
            self.val_aux_acc = MulticlassAccuracy(num_classes=aux_num_classes)
            self.test_aux_acc = MulticlassAccuracy(num_classes=aux_num_classes)
        else:
            self.train_aux_acc = BinaryAccuracy()
            self.val_aux_acc = BinaryAccuracy()
            self.test_aux_acc = BinaryAccuracy()
        
        self.train_aux_loss = MeanMetric()
        self.val_aux_loss = MeanMetric()
        self.test_aux_loss = MeanMetric()
        
        # Total loss metrics
        self.train_total_loss = MeanMetric()
        self.val_total_loss = MeanMetric()
        self.test_total_loss = MeanMetric()
        
        # Best auxiliary accuracy
        self.val_aux_acc_best = MaxMetric()

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
        
        # Auxiliary task loss criterion
        self.criterion_aux = torch.nn.CrossEntropyLoss()
        
        
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_aux_acc.reset()
        self.val_aux_loss.reset()
        self.val_aux_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor, main target labels, and auxiliary target labels.

        :return: A tuple containing (in order):
            - Main task loss.
            - Main task predictions.
            - Main task target labels.
            - Auxiliary task loss.
            - Auxiliary task predictions.
            - Auxiliary task target labels.
        """
        
        x, y, aux_y = batch
        logits, logits_aux = self.forward(x)
        
        # Main task
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        # Auxiliary task
        aux_loss = self.criterion_aux(logits_aux, aux_y)
        aux_preds = torch.argmax(logits_aux, dim=1)
        
        return loss, preds, y, aux_loss, aux_preds, aux_y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor, main labels, and auxiliary labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of combined losses between model predictions and targets.
        """
        loss, preds, targets, aux_loss, aux_preds, aux_targets = self.model_step(batch)

        # Combine losses
        total_loss = (1 - self.scale_aux) * loss + self.scale_aux * aux_loss
        
        # update and log main task metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/main_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/main_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # update and log auxiliary task metrics
        self.train_aux_loss(aux_loss)
        self.train_aux_acc(aux_preds, aux_targets)
        self.log("train/aux_loss", self.train_aux_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/aux_acc", self.train_aux_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # log total loss
        self.train_total_loss(total_loss)
        self.log("train/total_loss", self.train_total_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return total loss for backpropagation
        return total_loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor, main labels, and auxiliary labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, aux_loss, aux_preds, aux_targets = self.model_step(batch)

        # Combine losses
        total_loss = (1 - self.scale_aux) * loss + self.scale_aux * aux_loss
        
        # update and log main task metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/main_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/main_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # update and log auxiliary task metrics
        self.val_aux_loss(aux_loss)
        self.val_aux_acc(aux_preds, aux_targets)
        self.log("val/aux_loss", self.val_aux_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/aux_acc", self.val_aux_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # log total loss
        self.val_total_loss(total_loss)
        self.log("val/total_loss", self.val_total_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        self.log("val/main_acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        
        aux_acc = self.val_aux_acc.compute()  # get current val aux acc
        self.val_aux_acc_best(aux_acc)  # update best so far val aux acc
        self.log("val/aux_acc_best", self.val_aux_acc_best.compute(), sync_dist=True, prog_bar=True)
