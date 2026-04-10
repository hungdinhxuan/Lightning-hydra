#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fix-duration CNSL datamodule + MBCT collate (multi-band views).

Uses the same :class:`src.data.normal_datamodule.Dataset_for` /
``Dataset_for_dev`` pipeline as :class:`src.data.normal_datamodule.NormalDataModule`:
each sample is padded/cropped to ``trim_length`` in ``__getitem__``. Train and validation
dataloaders then apply :func:`src.data.components.collate_fn.mbct_collate_fn` to build
per-band batches for :class:`src.models.base.mbct_module.MBCTLitModule`.

By default ``mbct_max_length_sec`` is omitted in collate (``max_length_sec=None``) so lengths
are not adjusted twice; set ``mbct_fix_duration_in_dataset: false`` and ``mbct_max_length_sec``
if you want duration control only in collate instead.

For joint MDT (duration views) and MBCT, use
:class:`src.data.normal_mbct_mdt_datamodule.NormalMBCTMDTDataModule` and composite batch keys
like ``1_normal`` in ``weighted_views``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from torch.utils.data import DataLoader

from src.data.components.collate_fn import mbct_collate_fn
from src.data.normal_datamodule import NormalDataModule


class NormalMBCTDataModule(NormalDataModule):
    """Fix-duration data (see :class:`NormalDataModule`) + ``mbct_collate_fn`` on train/val."""

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        args: Optional[Dict[str, Any]] = None,
        chunking_eval: bool = False,
        enable_cache: bool = False,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            args=args,
            chunking_eval=chunking_eval,
            enable_cache=enable_cache,
        )
        self._setup_mbct_collate()

    def _setup_mbct_collate(self) -> None:
        a = self.args or {}
        wav_sr = int(a.get("wav_samp_rate", 16000))
        fix_in_ds = a.get("mbct_fix_duration_in_dataset", True)
        mbct_max_sec = a.get("mbct_max_length_sec", None)
        if fix_in_ds:
            collate_max_sec = None
        else:
            if mbct_max_sec is None:
                trim = a.get("trim_length", 64000)
                collate_max_sec = float(trim) / float(wav_sr)
            else:
                collate_max_sec = mbct_max_sec
        band_cfgs = a.get("mbct_band_configs", None)
        print(
            "[NormalMBCTDataModule] MBCT collate: "
            f"sample_rate={wav_sr}, max_length_sec={collate_max_sec}, "
            f"fix_duration_in_dataset={fix_in_ds}"
        )

        def _collate(batch):
            return mbct_collate_fn(
                batch,
                sample_rate=wav_sr,
                max_length_sec=collate_max_sec,
                padding_type=a.get("padding_type", "repeat"),
                random_start=a.get("random_start", False),
                band_configs=band_cfgs,
            )

        self.collate_fn = _collate

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
            persistent_workers=bool(self.hparams.num_workers),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=self.collate_fn,
            persistent_workers=bool(self.hparams.num_workers),
        )
