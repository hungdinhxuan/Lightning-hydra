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

class BaseLitModule(LightningModule):
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
        **kwargs,
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

        self.net = self.init_model(**kwargs)
        
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
            )
            self.net = peft.get_peft_model(self.net, lora_config)
            self.net.print_trainable_parameters()
        
        if lora_adapter_path is not None:
            self.load_lora_adapter(lora_adapter_path)
            
        # loss function
        cross_entropy_weight = torch.tensor(cross_entropy_weight)
        self.criterion = torch.nn.CrossEntropyLoss(cross_entropy_weight)

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.score_save_path = score_save_path

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

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
        pass

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        pass

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        pass

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        pass

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
