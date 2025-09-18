from typing import Any, Dict, Tuple
import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy
from typing import Union
from torch import nn
import torch
import os
import numpy as np
from src.utils import load_ln_model_weights


class BaseLitModule(LightningModule):

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.spec_eval = kwargs.get("spec_eval", False)
        self.score_save_path = kwargs.get("score_save_path", None)
        self.last_emb = kwargs.get("last_emb", False)
        self.emb_save_path = kwargs.get("emb_save_path", None)
        self.args = args
        
        # Create embedding save directory if specified
        if self.emb_save_path is not None:
            if not os.path.exists(self.emb_save_path):
                os.makedirs(self.emb_save_path)
        
        self.net = self.init_model(**kwargs)
        self.kwargs = kwargs
        # loss function
        
        self.criterion = self.init_criteria(**kwargs)
        
        # Optional: Initialize buffered writing for better performance
        self._write_buffer = []
        self._buffer_size = kwargs.get("write_buffer_size", 100)  # Buffer 100 batches by default
  
        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
    
    def init_criteria(self, **kwargs) -> torch.nn.Module:
        """
            Initialize the loss function with the given arguments. This method is used to initialize the loss
            function with the given arguments. The loss function is initialized with the given arguments and the
            loss function is returned.
            
            Base model doesn't implement this method. This method should be implemented in the derived
            model class.
        """
        raise NotImplementedError("init_criteria method is not implemented")

    def init_model(self, **kwargs) -> nn.Module:
        """
            Initialize the model with the given arguments. This method is used to initialize the model
            with the given arguments. The model is initialized with the given arguments and the model is
            returned.
            
            Base model doesn't implement this method. This method should be implemented in the derived
            model class.
        """
        raise NotImplementedError("init_model method is not implemented")

    def forward(self, x: torch.Tensor, inference_mode=False) -> torch.Tensor:
        return self.net(x) if not inference_mode else self.net(x)[0] # for inference mode, return only the first element

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
        if self.last_emb:
            self._export_embedding_file(batch)
        else:
            if self.score_save_path is not None:
                self._export_score_file(batch, batch_idx)
            else:
                raise ValueError("score_save_path is not provided")

    def _export_score_file(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, inference_mode=True) -> None:
        """Get the score file for the batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        batch_x, utt_id = batch
        
        # Forward pass
        batch_out = self.forward(batch_x, inference_mode=inference_mode)
        
        # Optimized tensor to numpy conversion (avoid .data and .tolist())
        if batch_out.is_cuda:
            scores_np = batch_out.detach().cpu().numpy()
        else:
            scores_np = batch_out.detach().numpy()
        
        # Pre-build all lines for batch writing (much faster than line-by-line)
        if self.spec_eval:
            batch_lines = [f'{fname} {scores[0]} {scores[1]}\n' 
                          for fname, scores in zip(utt_id, scores_np)]
        else:
            batch_lines = [f'{fname} {scores[1]}\n' 
                          for fname, scores in zip(utt_id, scores_np)]
        
        # Use buffered writing for maximum performance
        self._write_buffer.extend(batch_lines)
        
        # Flush buffer when it reaches the specified size
        if len(self._write_buffer) >= self._buffer_size * len(batch_lines):
            self._flush_buffer()

    def _flush_buffer(self):
        """Flush the write buffer to file."""
        if self._write_buffer:
            with open(self.score_save_path, 'a') as fh:
                fh.writelines(self._write_buffer)
            self._write_buffer.clear()

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Ensure any remaining buffered data is written
        self._flush_buffer()
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
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def _export_embedding_file(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """ Get the embedding file for the batch of data.
        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        """
        batch_x, utt_id = batch
        batch_emb = self.net(batch_x, last_emb=True)
        # import sys
        # print(f"Batch emb shape: {batch_emb.shape}")
        # sys.exit()
        fname_list = list(utt_id)

        for f, emb in zip(fname_list, batch_emb):
            f = f.split('/')[-1].split('.')[0]  # utt id only
            save_path_utt = os.path.join(self.emb_save_path, f)
            np.save(save_path_utt, emb.data.cpu().numpy())
