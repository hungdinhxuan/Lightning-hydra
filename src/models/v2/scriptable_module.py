import torch
from src.models.components.xlsr_vib import Model as XLSRVIB
from src.models.base.normal_module import NormalLitModule
from typing import Any, Dict, Tuple, Union
from torch import nn
import torch.nn.functional as F

class XLSRVIBNormalLitModule(NormalLitModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs  # 👈 Store kwargs for later use
        super().__init__(optimizer, scheduler, args, **kwargs)
        self.net = self.init_model(**kwargs)
        self.init_adapter()
        self.recon_weight_l = self.kwargs.get("recon_weight_l", 0.000001)
        self.recon_weight_b = self.kwargs.get("recon_weight_b", 0.05)
        
    def init_model(self, **kwargs) -> nn.Module:
        ssl_pretrained_path = kwargs.get("ssl_pretrained_path", None)
        if ssl_pretrained_path is None:
            raise ValueError("ssl_pretrained_path is required for XLSRVIBNormalLitModule")
        return XLSRVIB(ssl_pretrained_path)
    
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
      
        x, y = batch
        output, (decoded, mu, logvar, feats_w2v), emb = self.forward(x)
        
        # reconstruction loss
        BCE = F.binary_cross_entropy(torch.sigmoid(decoded), torch.sigmoid(feats_w2v), reduction='sum')
        
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        Recon_loss = self.recon_weight_l*(BCE + self.recon_weight_b*KLD)
        
        loss = self.criterion(output, y)
        preds = torch.argmax(output, dim=1)
        return loss, preds, y, Recon_loss
    
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, Recon_loss = self.model_step(batch)
        
        total_loss = loss + Recon_loss

        # update and log metrics
        self.train_loss(total_loss)
        self.train_acc(preds, targets)
        self.log("train/total_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/ce_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", Recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return total_loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, Recon_loss = self.model_step(batch)
        total_loss = loss + Recon_loss

        # update and log metrics
        self.val_loss(total_loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/ce_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon_loss", Recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss