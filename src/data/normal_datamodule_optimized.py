#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WebDataset-optimized replacement for normal datamodule."""

from __future__ import annotations

import os
import math
from typing import Any, Dict, Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader
from webdataset import WebLoader

from src.data.dataset_optimized import (
    build_wds_from_args,
    make_wds_collate_fn,
    read_protocol,
    split_entries_by_subset,
)


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


class NormalDataModule(LightningDataModule):
    """Drop-in datamodule using WebDataset shards instead of raw-file indexing."""

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
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.batch_size_per_device = batch_size
        self.data_dir = data_dir
        self.args = args or {}
        if _get_arg(self.args, "wds_data_dir", None) is None:
            self.args["wds_data_dir"] = os.path.join(self.data_dir, "wds_data")
        if _get_arg(self.args, "data_dir", None) is None:
            self.args["data_dir"] = self.data_dir

        self.chunking_eval = chunking_eval
        self.enable_cache = enable_cache

        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.collate_fn = make_wds_collate_fn(fallback_pad=True)
        self._subset_counts = None

    def _get_subset_counts(self) -> Dict[str, int]:
        if self._subset_counts is not None:
            return self._subset_counts

        protocol_path = _get_arg(self.args, "protocol_path", None)
        if not protocol_path or not os.path.exists(protocol_path):
            self._subset_counts = {"train": 0, "dev": 0, "eval": 0}
            return self._subset_counts

        entries = read_protocol(protocol_path)
        by_subset = split_entries_by_subset(entries)
        self._subset_counts = {
            "train": len(by_subset.get("train", [])),
            "dev": len(by_subset.get("dev", [])),
            "eval": len(by_subset.get("eval", [])),
        }
        return self._subset_counts

    def _auto_epoch_batches(self, split: str, *, drop_last: bool) -> Optional[int]:
        if not bool(_get_arg(self.args, "wds_auto_epoch_batches", False)):
            return None
        counts = self._get_subset_counts()
        n_samples = int(counts.get(split, 0))
        if n_samples <= 0:
            return None
        batch_size = int(self.batch_size_per_device)
        if batch_size <= 0:
            return None
        if drop_last:
            n_batches = n_samples // batch_size
        else:
            n_batches = int(math.ceil(float(n_samples) / float(batch_size)))
        return max(n_batches, 1)

    def _apply_epoch_batches(self, loader: WebLoader, split: str, *, drop_last: bool) -> WebLoader:
        key_map = {
            "train": "wds_train_epoch_batches",
            "dev": "wds_val_epoch_batches",
            "eval": "wds_test_epoch_batches",
        }
        key = key_map.get(split)
        if key is None:
            return loader
        n_batches = _get_arg(self.args, key, None)
        if n_batches is None:
            n_batches = self._auto_epoch_batches(split=split, drop_last=drop_last)
            if n_batches is None:
                return loader
        n_batches = int(n_batches)
        if n_batches <= 0:
            return loader
        return loader.with_epoch(n_batches)

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_train is None:
            self.data_train = build_wds_from_args(self.args, subset="train", include_eval_key=False)
            self.data_val = build_wds_from_args(self.args, subset="dev", include_eval_key=False)

            return_eval_key = bool(_get_arg(self.args, "wds_eval_return_utt_id", True))
            self.data_test = build_wds_from_args(self.args, subset="eval", include_eval_key=return_eval_key)

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

    def test_dataloader(self) -> DataLoader:
        batch_size = self.batch_size_per_device
        if bool(_get_arg(self.args, "no_pad", False)):
            batch_size = 1
        loader = WebLoader(
            self.data_test,
            batch_size=batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_fn,
            persistent_workers=bool(self.hparams.num_workers),
        )
        return self._apply_epoch_batches(loader, split="eval", drop_last=False)
