from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy

from typing import Union, List
import os
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch, omegaconf
from src.models.components.xlsr_conformertcm_baseline import Model as XLSRConformerTCM
from src.utils import load_ln_model_weights
from peft import LoraConfig, TaskType
import peft
from peft import PeftModel

class XLSRConformerTCMLoraLitModule(LightningModule):
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
        scheduler: torch.optim.lr_scheduler,
        optimizer: torch.optim.Optimizer,
        compile: Union[bool, None] = False,
        scheduler_tracking: Union[omegaconf.dictconfig.DictConfig, None] = None,
        args: Union[Dict[str, Any], None] = None,
        ssl_pretrained_path: str = None,
        score_save_path: str = None,
        cross_entropy_weight: List[float] = [0.1, 0.9],
        weighted_views: Dict[str, float] = {
            '1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0},
        adaptive_weights: bool = False,
        spec_eval: bool = False,
        base_line_ft_path: str = None,
        use_lora: bool = False,
        lora_adapter_path: str = None,
        last_emb: bool = False,
        emb_save_path: str = None,
        lora_adapter_paths: Union[List[str], str] = None,
        merge_adapters: bool = False,  # Whether to merge adapters
        adapter_weights: Union[List[float], None] = None,  # Weights for merging
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.use_lora = use_lora
        #self.lora_path = lora_path
        self.save_hyperparameters(logger=False)
        self.spec_eval = spec_eval
        self.last_emb = last_emb
        self.emb_save_path = emb_save_path
        
        if self.emb_save_path is not None:
            if not os.path.exists(self.emb_save_path):
                os.makedirs(self.emb_save_path)
                
        self.net = XLSRConformerTCM(args['conformer'], ssl_pretrained_path)
        
        if base_line_ft_path is not None:
            ckpt = torch.load(base_line_ft_path, weights_only=False)
            #args = ckpt['hyper_parameters']['args']['conformer']
            self.net = load_ln_model_weights(self.net, ckpt['state_dict'])
            print("Loaded baseline model from: ", base_line_ft_path)
        
        if self.use_lora:
            print("LoRA is enabled")
            lora_config = peft.LoraConfig(
                r=args['lora']['r'],
                target_modules=list(args['lora']['target_modules']),
                modules_to_save=list(args['lora']['modules_to_save']),
                lora_dropout=args['lora']['lora_dropout'], # Default 0.0
                lora_alpha=args['lora']['lora_alpha'], # Default 8
                init_lora_weights=args['lora'].get('init_lora_weights', True), # Passing True (default) results in the default initialization from the reference implementation from Microsoft, with the LoRA B weight being set to 0
            )
            self.net = peft.get_peft_model(self.net, lora_config)
            self.net.print_trainable_parameters()
        
        if lora_adapter_path is not None:
            self.load_lora_adapter(lora_adapter_path)
        
        if lora_adapter_paths is not None:
            # Convert single path to list for consistent handling
            if isinstance(lora_adapter_paths, str):
                lora_adapter_paths = [lora_adapter_paths]
                
            if self.merge_adapters and len(lora_adapter_paths) > 1:
                # If no weights provided, use equal weights
                if adapter_weights is None:
                    adapter_weights = [1.0/len(lora_adapter_paths)] * len(lora_adapter_paths)
                
                self.load_and_merge_adapters(lora_adapter_paths, adapter_weights)
            else:
                self.load_separate_adapters(lora_adapter_paths)
            
        # loss function
        cross_entropy_weight = torch.tensor(cross_entropy_weight)
        self.criterion = torch.nn.CrossEntropyLoss(cross_entropy_weight)

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.score_save_path = score_save_path
        
        print("weighted_views: ", weighted_views)

        self.train_view_acc = {
            view: BinaryAccuracy() for view in weighted_views
        }
        self.val_view_acc = {
            view: BinaryAccuracy() for view in weighted_views
        }
        self.test_view_acc = {}

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # for tracking best so far validation accuracy for each view
        self.val_view_acc_best = {
            view: MaxMetric() for view in weighted_views
        }

        self.train_loss_detail = {}
        self.val_loss_detail = {}
        self.running_loss = 0.0

        self.weighted_views = weighted_views
        self.adaptive_weights = adaptive_weights

        if self.adaptive_weights:
            self.weighted_views = {}
            for k, v in weighted_views.items():
                param = torch.nn.Parameter(
                    torch.tensor(float(v)), requires_grad=True)
                self.register_parameter(f"adaptive_weight_{k}", param)
                self.weighted_views[k] = param

    def load_separate_adapters(self, adapter_paths):
        """Load multiple LoRA adapters separately.
        
        :param adapter_paths: List of paths to LoRA adapters
        """
        for i, adapter_path in enumerate(adapter_paths):
            adapter_name = f"adapter_{i}"
            self.net.load_adapter(adapter_path, adapter_name=adapter_name)
            print(f"Loaded LoRA adapter from: {adapter_path} as {adapter_name}")
        
        # Set the first one as active by default
        if adapter_paths:
            self.net.set_adapter(adapter_name="adapter_0")
            print("Set adapter_0 as the active adapter")
    
    def load_and_merge_adapters(self, adapter_paths, weights):
        """Load and merge multiple LoRA adapters.
        
        :param adapter_paths: List of paths to LoRA adapters
        :param weights: List of weights for each adapter in the merge
        """
        # First, load each adapter with a unique name
        adapter_names = []
        for i, adapter_path in enumerate(adapter_paths):
            adapter_name = f"adapter_{i}"
            adapter_names.append(adapter_name)
            self.net.load_adapter(adapter_path, adapter_name=adapter_name)
            print(f"Loaded LoRA adapter from: {adapter_path} as {adapter_name}")
        
        # Merge the adapters
        merged_adapter_name = "merged_adapter"
        self.net.add_adapter(merged_adapter_name, adapter_type="lora")
        
        # Create a dictionary of weights for merging
        weight_dict = {name: weight for name, weight in zip(adapter_names, weights)}
        
        # Perform the merge operation
        self.net.merge_and_unload(adapter_names=adapter_names, 
                                 weights=weight_dict,
                                 adapter_name=merged_adapter_name)
        
        # Set the merged adapter as active
        self.net.set_adapter(adapter_name=merged_adapter_name)
        print(f"Merged {len(adapter_paths)} adapters with weights {weights} as '{merged_adapter_name}'")
    
    def set_active_adapter(self, adapter_idx):
        """Set the active adapter by index (for non-merged mode).
        
        :param adapter_idx: Index of the adapter to activate
        """
        if self.merge_adapters:
            print("Warning: In merged adapter mode. Using the merged adapter.")
            self.net.set_adapter(adapter_name="merged_adapter")
        else:
            adapter_name = f"adapter_{adapter_idx}"
            self.net.set_adapter(adapter_name=adapter_name)
            print(f"Set {adapter_name} as the active adapter")
            
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
        if self.last_emb:
            self._export_embedding_file(batch)
            # print("Embedding file saved")
        else:
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
                
    def _export_embedding_file(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """ Get the embedding file for the batch of data.
        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        """
        batch_x, utt_id = batch
        batch_emb = self.net(batch_x, last_emb=True)

        fname_list = list(utt_id)

        for f, emb in zip(fname_list, batch_emb):
            f = f.split('/')[-1].split('.')[0]  # utt id only
            save_path_utt = os.path.join(self.emb_save_path, f)
            np.save(save_path_utt, emb.data.cpu().numpy())

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
        optimizer = self.hparams.optimizer(
            params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            if self.hparams.scheduler_tracking is not None:
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": self.hparams.scheduler_tracking.monitor,
                        "interval": self.hparams.scheduler_tracking.interval,
                        "frequency": self.hparams.scheduler_tracking.frequency,
                    },
                }
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


# if __name__ == "__main__":
#     _ = WAVLMVIBLLitModule(None, None, None, None)
