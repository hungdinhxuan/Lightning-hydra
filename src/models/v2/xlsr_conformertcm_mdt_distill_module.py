from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch
from torchmetrics import MeanMetric
from torchmetrics.classification.accuracy import BinaryAccuracy

from src.models.components.continual_distill_strategy import ContinualDistillStrategy
from src.models.components.distillation import FeatureKDLoss, LogitKDLoss
from src.models.components.teacher_wrapper import load_frozen_teacher_model
from src.models.v2.xlsr_conformertcm_mdt_module import XLSRConformertcmMDTLitModule


class XLSRConformertcmMDTDistillLitModule(XLSRConformertcmMDTLitModule):
    """MDT LoRA module extended with optional teacher-student distillation."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        args: Union[Dict[str, Any], None] = None,
        distill: Union[Dict[str, Any], None] = None,
        **kwargs,
    ) -> None:
        self.distill_cfg = dict(distill or {})
        super().__init__(optimizer, scheduler, args, distill=distill, **kwargs)

        self.distill_enabled = bool(self.distill_cfg.get("enabled", False))
        self.distill_temperature = float(self.distill_cfg.get("temperature", 4.0))
        self.lambda_kd = float(self.distill_cfg.get("lambda_kd", 0.3))
        self.lambda_feat = float(self.distill_cfg.get("lambda_feat", 0.0))
        self.distill_strategy = ContinualDistillStrategy(
            apply_on=self.distill_cfg.get("apply_on", "replay_only")
        )
        self.logit_kd_loss = LogitKDLoss(temperature=self.distill_temperature)
        self.feature_kd_loss = FeatureKDLoss()

        self.train_mdt_loss = MeanMetric()
        self.train_kd_loss = MeanMetric()
        self.train_feat_kd_loss = MeanMetric()

        self.teacher_model = None
        if self.distill_enabled:
            teacher_ckpt_path = self.distill_cfg.get("teacher_ckpt_path")
            if not teacher_ckpt_path:
                raise ValueError("distill.teacher_ckpt_path is required when distill.enabled=true")
            self.teacher_model = load_frozen_teacher_model(
                model_factory=lambda: self.init_model(**kwargs),
                checkpoint_path=teacher_ckpt_path,
                checkpoint_format=self.distill_cfg.get("checkpoint_format", "auto"),
                strict=bool(self.distill_cfg.get("strict", True)),
            )

    def on_train_start(self) -> None:
        super().on_train_start()
        if self.teacher_model is not None:
            self.teacher_model.eval()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets, loss_detail, view_acc, kd_loss, feat_kd_loss = self._distill_model_step(batch)
        total_loss = loss + self.lambda_kd * kd_loss + self.lambda_feat * feat_kd_loss

        self._update_train_view_metrics(loss_detail, view_acc)
        self.train_loss(total_loss)
        self.train_mdt_loss(loss)
        self.train_kd_loss(kd_loss)
        self.train_feat_kd_loss(feat_kd_loss)
        self.train_acc(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("train/mdt_loss", self.train_mdt_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("train/ce_loss", self.train_mdt_loss, on_step=False, on_epoch=True,
                 prog_bar=False, sync_dist=True, batch_size=self.batch_size)
        self.log("train/kd_loss", self.train_kd_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("train/feat_kd_loss", self.train_feat_kd_loss, on_step=False, on_epoch=True,
                 prog_bar=False, sync_dist=True, batch_size=self.batch_size)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets, loss_detail, view_acc = self._supervised_multiview_step(batch)

        for k, v in view_acc.items():
            self.val_view_acc[k] = self.val_view_acc.get(k, BinaryAccuracy())
            _preds, _targets = v[0]
            self.val_view_acc[k](_preds, _targets)

        for k, v in loss_detail.items():
            self.val_loss_detail[k] = self.val_loss_detail.get(k, MeanMetric())
            self.val_loss_detail[k](v)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True, batch_size=self.batch_size)

    def _distill_model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        loss, preds, targets, loss_detail, view_acc, student_outputs = self._supervised_multiview_step(
            batch,
            return_outputs=True,
        )

        kd_loss = loss.new_zeros(())
        feat_kd_loss = loss.new_zeros(())
        if not self.distill_enabled:
            return loss, preds, targets, loss_detail, view_acc, kd_loss, feat_kd_loss
        if self.teacher_model is None:
            raise RuntimeError("distillation enabled but teacher_model is not initialized")

        for view, (x, _y, sources, student_logits) in student_outputs.items():
            with torch.no_grad():
                teacher_logits = self.teacher_model(x)
            kd_mask = self.distill_strategy.build_mask(
                sources=sources,
                batch_size=x.size(0),
                device=x.device,
            )
            kd_loss = kd_loss + self.logit_kd_loss(
                student_logits[kd_mask],
                teacher_logits[kd_mask],
                temperature=self.distill_temperature,
            ) * self.weighted_views[str(view)]

        return loss, preds, targets, loss_detail, view_acc, kd_loss, feat_kd_loss

    def _supervised_multiview_step(self, batch, return_outputs: bool = False):
        train_loss = 0.0
        all_preds = []
        all_labels = []
        loss_detail = {}
        view_acc = {}
        student_outputs = {}

        for view, batch_item in batch.items():
            x, y, sources = self._unpack_view_batch(batch_item)
            self.batch_size = x.size(0) * len(self.weighted_views.keys())
            view = str(view)

            if x.dtype != torch.float32:
                x = x.float()

            logits = self.forward(x)
            loss = self.criterion(logits, y) * self.weighted_views[view]
            train_loss += loss
            preds = torch.argmax(logits, dim=1)

            view_acc[view] = view_acc.get(view, [])
            view_acc[view].append([preds.detach().cpu(), y.detach().cpu()])
            all_preds.append(preds)
            all_labels.append(y)
            loss_detail[view] = loss_detail.get(view, 0) + loss.item()
            student_outputs[view] = (x, y, sources, logits)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        if return_outputs:
            return train_loss, all_preds, all_labels, loss_detail, view_acc, student_outputs
        return train_loss, all_preds, all_labels, loss_detail, view_acc

    @staticmethod
    def _unpack_view_batch(batch_item):
        if len(batch_item) == 2:
            x, y = batch_item
            return x, y, None
        if len(batch_item) == 3:
            x, y, sources = batch_item
            return x, y, sources
        raise ValueError(f"expected view batch item of length 2 or 3, got {len(batch_item)}")

    def _update_train_view_metrics(self, loss_detail, view_acc) -> None:
        for k, v in view_acc.items():
            self.train_view_acc[k] = self.train_view_acc.get(k, BinaryAccuracy())
            _preds, _targets = v[0]
            self.train_view_acc[k](_preds, _targets)

        for k, v in loss_detail.items():
            self.train_loss_detail[k] = self.train_loss_detail.get(k, MeanMetric())
            self.train_loss_detail[k](v)
