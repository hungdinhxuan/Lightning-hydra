from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy

import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
from src.models.components.wavlmbase_vib import Model as WavlmBaseVIB
from src.utils.debug import NaNErrorMode
from src.models.components.xlsr_conformertcm import Model as XLSRConformerTCM
from src.metrics.eer import EERMetric

class XLSRConformerTCMLitModule(LightningModule):
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
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        args: Union[Dict[str, Any], None] = None,
        cp_path: str = None,
        is_train: bool = True,
        score_save_path: str = None,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = XLSRConformerTCM(args, cp_path, is_train)

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.score_save_path = score_save_path
        #self.test_eer = EERMetric()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.args = args
        self.is_train = is_train

        # for optimizer and scheduler
        self.running_loss = 0.0
        self.num_total = 0.0
        self.train_loss_detail = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

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
        
        train_loss = 0.0
        OC_info = {
            'previous_C': None,
            'previous_num_bona': None
        }   
        
        info, batch_x, batch_y = batch
        
        if len(batch_x.shape) == 3:
            batch_x = batch_x.squeeze(0).transpose(0, 1)

        batch_y = batch_y.view(-1).type(torch.int64)
        # print('batch_y', batch_y.shape)
        self.num_total += batch_y.shape[0]

        with NaNErrorMode(
            enabled=False, raise_error=False, print_stats=True, print_nan_index=False
        ):
            batch_out, batch_feat, batch_emb = self.net(batch_x)
            # losses = loss_custom(batch_out, batch_feat, batch_emb, batch_y, config)
            # OC for the OC loss
            losses = self.net.loss(batch_out, batch_feat,
                                batch_emb, batch_y, self.args, OC_info)

            for key, value in losses.items():
                train_loss += value
                self.train_loss_detail[key] = self.train_loss_detail.get(
                    key, 0) + value.item()

        self.running_loss += train_loss.item()
        _, preds = batch_out.max(dim=1)
        #preds = torch.argmax(batch_out, dim=1)
        #return loss, preds, y
        return train_loss, preds, batch_y, self.train_loss_detail

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, train_loss_detail = self.model_step(batch)

        # update and log metrics
        self.train_loss(self.running_loss) 
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        for loss_name, _loss in train_loss_detail.items():
            self.log(f"train/{loss_name}", _loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."

        # Reset metrics at the end of the epoch
        self.running_loss = 0.0
        self.num_total = 0.0
        self.train_loss_detail = {}
        

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, val_loss_detail = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for loss_name, _loss in val_loss_detail.items():
            self.log(f"train/{loss_name}", _loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        batch_x, utt_id = batch
        batch_out = self.net(batch_x)

        # update and log metrics
        # self.test_loss(loss)
        # self.test_acc(preds, targets)
        # self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        # Write scores to file
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
        # eer = self.test_eer.compute()
        # self.log("test/eer", eer, sync_dist=True, prog_bar=True)
        # self.test_eer.reset()
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # if self.hparams.compile and stage == "fit":
        #     self.net = torch.compile(self.net)
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
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
#     _ = WAVLMVIBLLitModule(None, None, None, None)