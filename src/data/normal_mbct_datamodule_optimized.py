#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WebDataset-optimized MBCT datamodule."""

from __future__ import annotations

from typing import Any, Dict, Optional

from torch.utils.data import DataLoader
from webdataset import WebLoader

from src.data.components.collate_fn import mbct_collate_fn
from src.data.normal_datamodule_optimized import NormalDataModule


def _get_arg(args: Any, key: str, default: Any = None) -> Any:
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(key, default)
    if hasattr(args, "get"):
        try:
            return args.get(key, default)
        except Exception:
            pass
    return getattr(args, key, default)


class NormalMBCTDataModule(NormalDataModule):
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
        wav_sr = int(_get_arg(a, "wav_samp_rate", 16000))
        fix_in_ds = bool(_get_arg(a, "mbct_fix_duration_in_dataset", True))
        mbct_max_sec = _get_arg(a, "mbct_max_length_sec", None)
        if fix_in_ds:
            collate_max_sec = None
        else:
            if mbct_max_sec is None:
                trim = _get_arg(a, "trim_length", 64000)
                collate_max_sec = float(trim) / float(wav_sr)
            else:
                collate_max_sec = mbct_max_sec
        band_cfgs = _get_arg(a, "mbct_band_configs", None)

        def _collate(batch):
            return mbct_collate_fn(
                batch,
                sample_rate=wav_sr,
                max_length_sec=collate_max_sec,
                padding_type=_get_arg(a, "padding_type", "repeat"),
                random_start=bool(_get_arg(a, "random_start", False)),
                band_configs=band_cfgs,
            )

        self.collate_fn = _collate

    def train_dataloader(self) -> DataLoader:
        loader = WebLoader(
            self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            drop_last=True,
            persistent_workers=bool(self.hparams.num_workers),
        )
        return self._apply_epoch_batches(loader, split="train", drop_last=True)

    def val_dataloader(self) -> DataLoader:
        loader = WebLoader(
            self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=bool(self.hparams.num_workers),
        )
        return self._apply_epoch_batches(loader, split="dev", drop_last=False)
