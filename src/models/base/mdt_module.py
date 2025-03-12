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
from src.models.base.base_module import BaseLitModule

class MDTModule(BaseLitModule):
  
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
     
        super().__init__()
        
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
        self.running_loss = 0.0

        self.weighted_views = weighted_views
        self.net = self.init_model(**kwargs)
        

    def init_model(self, **kwargs) -> nn.Module:
        """
            Initialize the model with the given arguments. This method is used to initialize the model
            with the given arguments. The model is initialized with the given arguments and the model is
            returned.
            
            Base model doesn't implement this method. This method should be implemented in the derived
            model class.
        """
        raise NotImplementedError("init_model method is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

        # Reset all details for loss and accuracy

        for k, v in self.val_loss_detail.items():
            self.val_loss_detail[k].reset()

        for k, v in self.val_view_acc.items():
            self.val_view_acc[k].reset()

        for k, v in self.val_view_acc_best.items():
            self.val_view_acc_best[k].reset()

        # Log current adaptive_weights
        if self.adaptive_weights:
            print("Adaptive weights are enabled")

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

            # Update view accuracy metric to view_acc
            # Initialize view accuracy metric
            view_acc[view] = view_acc.get(view, [])

            # Append predictions and target to view_acc
            view_acc[view].append([preds.cpu(), y.cpu()])

            # Collect predictions and labels
            all_preds.append(preds)
            all_labels.append(y)

            # Intialize loss_detail dictionary
            loss_detail[view] = loss_detail.get(view, 0) + loss.item()

        # self.running_loss += train_loss.item()

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

        self.running_loss += loss.item()

        # Update train_view_acc and log metrics
        for k, v in view_acc.items():
            # Initialize train_view_acc dictionary
            self.train_view_acc[k] = self.train_view_acc.get(
                k, BinaryAccuracy())
            _preds, _targets = v[0]
            # v[0] is preds, v[1] is targets
            self.train_view_acc[k](_preds, _targets)
            # self.log(f"train/view_{k}_acc", self.train_view_acc[k].compute(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Update train_loss_detail and log metrics
        for k, v in loss_detail.items():
            # Initialize train_loss_detail dictionary
            self.train_loss_detail[k] = self.train_loss_detail.get(
                k, MeanMetric())
            self.train_loss_detail[k](v)

            # self.train_loss_detail[k] = self.train_loss_detail.get(k, 0) + v
            # self.log(f"train/view_{k}_loss", self.train_loss_detail[k].compute(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Update and log train loss and accuracy
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        self.log_dict({f"adaptive_weight_{k}": v for k, v in self.weighted_views.items(
        )}, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
            # v[0] is preds, v[1] is targets
            self.val_view_acc[k](_preds, _targets)
            # self.log(f"val/view_{k}_acc", self.val_view_acc[k].compute(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Update val_loss_detail and log metrics
        for k, v in loss_detail.items():
            # Initialize val_loss_detail dictionary
            self.val_loss_detail[k] = self.val_loss_detail.get(k, MeanMetric())
            self.val_loss_detail[k](v)
            # self.log(f"val/view_{k}_loss", self.val_loss_detail[k].compute(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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

        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(),
                 sync_dist=True, prog_bar=True, batch_size=self.batch_size)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        if self.score_save_path is not None:
            self._export_score_file(batch)
        else:
            raise ValueError("score_save_path is not provided")

    def _export_score_file(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Get the score file for the batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        """
        batch_x, utt_id = batch
        batch_out = self.net(batch_x)

        fname_list = list(utt_id)
        score_list = batch_out.data.cpu().numpy().tolist()

        with open(self.score_save_path, 'a+') as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write('{} {} {}\n'.format(f, cm[0], cm[1])) if self.spec_eval else fh.write(
                    '{} {}\n'.format(f, cm[1]))

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(
            params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
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
    
    def load_lora_adapter(self, checkpoint_path: str, adapter_name: str = "default"):
        """Specialized method for loading LoRA adapters"""
        if hasattr(self.net, 'load_adapter'):
            self.net.load_adapter(checkpoint_path, adapter_name=adapter_name)
            self.net.set_adapter(adapter_name)
        else:
            self.net = PeftModel.from_pretrained(self.net, checkpoint_path)
            self.net.merge_and_unload()
        
        print(f"Loaded LoRA adapter from {checkpoint_path}")
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)
