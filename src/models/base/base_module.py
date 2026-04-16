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
import threading
import queue
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
        
        # Stable buffered text writing (low memory footprint, less open/close overhead).
        self._write_buffer = []
        # If write_buffer_size is unset, adapt flush cadence to current batch size.
        self._buffer_size = kwargs.get("write_buffer_size", None)  # number of batches before flush
        self._buffer_target_rows = kwargs.get("write_buffer_target_rows", 65536)
        self._dynamic_flush_batches = 32
        self._buffered_batches = 0
        self._score_fh = None
        self._async_score_write = kwargs.get("async_score_write", True)
        self._score_writer_queue_size = kwargs.get("score_writer_queue_size", 16)
        self._score_writer_queue = None
        self._score_writer_thread = None
        self._score_writer_error = None
  
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

    @torch.no_grad()
    @torch.inference_mode(True)
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

    def on_test_start(self) -> None:
        """Lightning hook that is called when a test epoch starts."""
        self._write_buffer.clear()
        self._buffered_batches = 0
        self._score_writer_error = None
        if self._score_fh is not None:
            self._score_fh.close()
            self._score_fh = None
        if self.score_save_path is not None:
            self._score_fh = open(self.score_save_path, 'a', encoding='utf-8')
        if self._async_score_write:
            self._score_writer_queue = queue.Queue(maxsize=max(1, self._score_writer_queue_size))
            self._score_writer_thread = threading.Thread(
                target=self._score_writer_worker,
                name="score-writer-thread",
                daemon=True,
            )
            self._score_writer_thread.start()

    def _score_writer_worker(self) -> None:
        """Background worker: formats and writes score chunks."""
        try:
            while True:
                item = self._score_writer_queue.get()
                if item is None:
                    self._score_writer_queue.task_done()
                    break

                utt_chunk, score_chunk = item
                if self.spec_eval:
                    self._score_fh.writelines(
                        f"{fname} {scores[0]} {scores[1]}\n"
                        for fname, scores in zip(utt_chunk, score_chunk)
                    )
                else:
                    self._score_fh.writelines(
                        f"{fname} {score}\n"
                        for fname, score in zip(utt_chunk, score_chunk)
                    )
                self._score_writer_queue.task_done()
        except Exception as exc:
            self._score_writer_error = exc

    def _export_score_file(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int, inference_mode=True) -> None:
        """Get the score file for the batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        batch_x, utt_id = batch
        
        # Forward pass
        is_jit_model = self.kwargs.get("is_jit_model", False)
        if is_jit_model:
            batch_out = self.net(batch_x)
        else:
            batch_out = self.forward(batch_x, inference_mode=inference_mode)
        
        # Reduce GPU->CPU transfer volume by slicing needed scores on GPU first.
        batch_out_detached = batch_out.detach()
        if self.spec_eval:
            # Keep both class scores for the expected text output format.
            score_tensor = batch_out_detached[:, :2]
        else:
            # In normal mode we only need score index 1.
            score_tensor = batch_out_detached[:, 1]
        score_chunk = np.ascontiguousarray(score_tensor.to("cpu").numpy())

        if self._async_score_write:
            if self._score_writer_error is not None:
                raise RuntimeError(f"Background score writer failed: {self._score_writer_error}")
            self._score_writer_queue.put((list(utt_id), score_chunk))
        else:
            if self.spec_eval:
                batch_lines = [
                    f"{fname} {scores[0]} {scores[1]}\n"
                    for fname, scores in zip(utt_id, score_chunk)
                ]
            else:
                batch_lines = [
                    f"{fname} {score}\n"
                    for fname, score in zip(utt_id, score_chunk)
                ]

            self._write_buffer.extend(batch_lines)
            if self._buffer_size is None:
                batch_rows = max(1, len(batch_lines))
                self._dynamic_flush_batches = max(1, self._buffer_target_rows // batch_rows)
            self._buffered_batches += 1
            flush_every_batches = self._buffer_size if self._buffer_size is not None else self._dynamic_flush_batches
            if self._buffered_batches >= flush_every_batches:
                self._flush_buffer()

    def _flush_buffer(self):
        """Flush buffered lines to score file."""
        if not self._write_buffer:
            return
        if self._score_fh is None:
            with open(self.score_save_path, 'a', encoding='utf-8') as fh:
                fh.writelines(self._write_buffer)
        else:
            self._score_fh.writelines(self._write_buffer)
        self._write_buffer.clear()
        self._buffered_batches = 0

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        if self._async_score_write and self._score_writer_queue is not None:
            self._score_writer_queue.put(None)
            self._score_writer_queue.join()
            if self._score_writer_thread is not None:
                self._score_writer_thread.join(timeout=5.0)
            self._score_writer_queue = None
            self._score_writer_thread = None
            if self._score_writer_error is not None:
                raise RuntimeError(f"Background score writer failed: {self._score_writer_error}")
        else:
            self._flush_buffer()
        if self._score_fh is not None:
            self._score_fh.flush()
            self._score_fh.close()
            self._score_fh = None
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        # print(f"Compiling model in setup hook for stage: {stage}")
        # if self.hparams.compile and stage == "fit":
        #     self.net = torch.compile(self.net, dynamic=False)
        #     print("Model compiled!")
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
