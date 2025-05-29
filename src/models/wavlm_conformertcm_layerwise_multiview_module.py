from typing import Any, Dict, Tuple, Union, List

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy
from src.models.components.wavlm_layerwise_conformertcm import Model as WavlmConformerTCM

class WavlmConformerTCMLitModule(LightningModule):
    """LightningModule for WavLM-Conformer-TCM model with layerwise features and multiview support.
    
    This module implements a WavLM-Conformer-TCM model that:
    1. Uses layerwise features from WavLM
    2. Supports multiple views of the input data
    3. Allows adaptive weighting of different views
    4. Implements training, validation and testing steps
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        ssl_pretrained_path: str = None,
        score_save_path: str = None,
        cross_entropy_weight: list[float] = [0.1, 0.9],
        weighted_views: Dict[str, float] = {
            '1': 1.0, '2': 1.0, '3': 1.0, '4': 1.0},
        adaptive_weights: bool = False,
        spec_eval: bool = False,
        n_layers: int = 24,
        ssl_freeze: bool = False,
    ) -> None:
        """Initialize the WavLM-Conformer-TCM module.

        Args:
            optimizer: The optimizer to use for training
            scheduler: The learning rate scheduler to use for training
            args: Configuration arguments for the model
            ssl_pretrained_path: Path to pretrained SSL model weights
            score_save_path: Path to save prediction scores
            cross_entropy_weight: Weights for cross entropy loss
            weighted_views: Weights for different views
            adaptive_weights: Whether to use adaptive view weights
            spec_eval: Whether to use special evaluation mode
            n_layers: Number of WavLM layers to use
            ssl_freeze: Whether to freeze SSL model weights
        """
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.spec_eval = spec_eval

        self.net = WavlmConformerTCM(args['conformer'], ssl_pretrained_path, n_layers)
        
        if ssl_freeze:
            self.net.ssl_model.freeze_model()
            
        # loss function
        cross_entropy_weight = torch.tensor(cross_entropy_weight)
        self.criterion = torch.nn.CrossEntropyLoss(cross_entropy_weight)

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()
        self.score_save_path = score_save_path

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
        self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, List]]:
        """Perform a single model step on a batch of data.

        Args:
            batch: A dictionary mapping view keys to tuples of (input_tensor, target_tensor)

        Returns:
            Tuple containing:
            - Total loss tensor
            - Concatenated predictions tensor
            - Concatenated target labels tensor
            - Dictionary of per-view losses
            - Dictionary of per-view accuracy metrics
        """
        train_loss = 0.0
        all_preds = []
        all_labels = []
        loss_detail = {}
        view_acc = {}

        for view, (x, y) in batch.items():
            view = str(view)  # Convert view to string for indexing
            self.batch_size = x.size(0) * len(self.weighted_views.keys())

            # Forward pass
            logits = self.forward(x.float())
            loss = self.criterion(logits, y) * self.weighted_views[view]
            train_loss += loss
            preds = torch.argmax(logits, dim=1)

            # Store predictions and labels
            all_preds.append(preds)
            all_labels.append(y)
            
            # Update metrics
            view_acc[view] = [[preds.cpu(), y.cpu()]]
            loss_detail[view] = loss.item()

        # Concatenate predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        return train_loss, all_preds, all_labels, loss_detail, view_acc

    def training_step(
        self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data.

        Args:
            batch: A dictionary mapping view keys to tuples of (input_tensor, target_tensor)
            batch_idx: The index of the current batch

        Returns:
            Loss tensor for backpropagation
        """
        loss, preds, targets, loss_detail, view_acc = self.model_step(batch)
        self.running_loss += loss.item()

        # Update metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)

        # Update per-view metrics
        for view, (view_preds, view_targets) in view_acc.items():
            self.train_view_acc[view](view_preds, view_targets)
            self.train_loss_detail[view] = self.train_loss_detail.get(view, MeanMetric())
            self.train_loss_detail[view](loss_detail[view])

        # Log metrics
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)

        # Log adaptive weights if enabled
        if self.adaptive_weights:
            self.log_dict({f"adaptive_weight_{k}": v for k, v in self.weighted_views.items()},
                         on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook that is called when a training epoch ends."""
        # Log per-view metrics
        for view in self.train_view_acc:
            self.log(f"train/view_{view}_acc", self.train_view_acc[view].compute(),
                     prog_bar=True, sync_dist=True)
            self.log(f"train/view_{view}_loss", self.train_loss_detail[view].compute(),
                     prog_bar=True, sync_dist=True)

        # Log adaptive weights if enabled
        if self.adaptive_weights:
            self.log_dict({f"adaptive_weight_{k}": v for k, v in self.weighted_views.items()},
                         on_epoch=True, prog_bar=True, sync_dist=True)

    def validation_step(
        self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> None:
        """Perform a single validation step on a batch of data.

        Args:
            batch: A dictionary mapping view keys to tuples of (input_tensor, target_tensor)
            batch_idx: The index of the current batch
        """
        loss, preds, targets, loss_detail, view_acc = self.model_step(batch)

        # Update metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)

        # Update per-view metrics
        for view, (view_preds, view_targets) in view_acc.items():
            self.val_view_acc[view](view_preds, view_targets)
            self.val_loss_detail[view] = self.val_loss_detail.get(view, MeanMetric())
            self.val_loss_detail[view](loss_detail[view])

        # Log metrics
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)

    def on_validation_epoch_end(self) -> None:
        """Lightning hook that is called when a validation epoch ends."""
        # Update and log best validation accuracy
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(),
                 sync_dist=True, prog_bar=True, batch_size=self.batch_size)

        # Log per-view metrics
        for view in self.val_view_acc:
            acc = self.val_view_acc[view].compute()
            self.val_view_acc_best[view](acc)
            self.log(f"val/view_{view}_acc", acc, prog_bar=True, sync_dist=True)
            self.log(f"val/view_{view}_acc_best", self.val_view_acc_best[view].compute(),
                     prog_bar=True, sync_dist=True)
            self.log(f"val/view_{view}_loss", self.val_loss_detail[view].compute(),
                     prog_bar=True, sync_dist=True)

    def test_step(
        self, batch: Dict[str, Tuple[torch.Tensor, torch.Tensor]], batch_idx: int
    ) -> None:
        """Perform a single test step on a batch of data.

        Args:
            batch: A dictionary mapping view keys to tuples of (input_tensor, target_tensor)
            batch_idx: The index of the current batch
        """
        if self.score_save_path is None:
            raise ValueError("score_save_path must be provided for testing")
            
        loss, preds, targets, loss_detail, view_acc = self.model_step(batch)
        
        # Update metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        
        # Update per-view metrics
        for view, (view_preds, view_targets) in view_acc.items():
            self.test_view_acc[view] = self.test_view_acc.get(view, BinaryAccuracy())
            self.test_view_acc[view](view_preds, view_targets)
            
        # Log metrics
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        # Log per-view metrics
        for view in self.test_view_acc:
            self.log(f"test/view_{view}_acc", self.test_view_acc[view].compute(),
                     prog_bar=True, sync_dist=True)

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit, validate, test, or predict.

        Args:
            stage: Either "fit", "validate", "test", or "predict"
        """
        pass

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dictionary containing optimizer and optional scheduler configuration
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}
            
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

    def optimizer_zero_grad(self, epoch: int, batch_idx: int, optimizer: torch.optim.Optimizer) -> None:
        """Zero out gradients with improved memory efficiency.

        Args:
            epoch: Current epoch number
            batch_idx: Current batch index
            optimizer: The optimizer to zero gradients for
        """
        optimizer.zero_grad(set_to_none=True)


# if __name__ == "__main__":
#     _ = WAVLMVIBLLitModule(None, None, None, None)
