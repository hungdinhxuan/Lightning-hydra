from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy

from typing import Union
from torchmetrics.metric import Metric
import torch
from src.models.components.xlsr_conformertcm_baseline import Model as XLSRConformerTCM
from src.models.base.base_module import BaseLitModule

class TeacherStudentLitModule(BaseLitModule):
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(optimizer, scheduler, args, **kwargs)
        # Initialize metrics
        self.init_criteria(**kwargs)
        self.teacher_net = self.t_init_model(**kwargs)
        self.teacher_net.eval()
        # self.teacher_net.requires_grad = False
        # self.teacher_net.freeze()
        self.losses = self.init_losses(**kwargs)
    
    def init_losses(self, **kwargs) -> Dict[str, Metric]:
        """
            Initialize the losses with the given arguments. This method is used to initialize the losses
            with the given arguments. The losses are initialized with the given arguments and the losses are returned.
        """
        raise NotImplementedError("init_losses method is not implemented")

    def t_init_model(self, **kwargs) -> torch.nn.Module:
        """
            Initialize the teacher model with the given arguments. This method is used to initialize the teacher model
            with the given arguments. The teacher model is initialized with the given arguments and the teacher model is returned.
        """
        raise NotImplementedError("t_init_model method is not implemented")

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
    
    def teacher_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single teacher step on a batch of data.
        """
        x, y = batch
        logits = self.teacher_net(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y
        
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
            - A dictionary of detailed losses.
        """
        
        # x, y = batch
        # logits = self.forward(x)
        # loss = self.criterion(logits, y)
        # preds = torch.argmax(logits, dim=1)
        # return loss, preds, y
        raise NotImplementedError("model_step method is not implemented")

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        raise NotImplementedError("training_step method is not implemented")

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
