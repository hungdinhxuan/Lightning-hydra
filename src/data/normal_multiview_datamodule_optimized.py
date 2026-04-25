#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WebDataset-optimized MDT-only datamodule."""

from __future__ import annotations

from typing import Any, Dict, Optional

from torch.utils.data import DataLoader
from webdataset import WebLoader

from src.data.components.collate_fn import (
    ChunkingCollator,
    multi_view_collate_fn,
    variable_multi_view_collate_fn,
)
from src.data.normal_datamodule_optimized import NormalDataModule as BaseNormalDataModule


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


class NormalDataModule(BaseNormalDataModule):
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
        self._setup_mv_collate()

    def _setup_mv_collate(self) -> None:
        a = self.args or {}
        is_variable = bool(_get_arg(a, "is_variable_multi_view", False))

        if is_variable:
            top_k = int(_get_arg(a, "top_k", 4))
            min_duration = int(_get_arg(a, "min_duration", 16000))
            max_duration = int(_get_arg(a, "max_duration", 64000))
            wav_sr = int(_get_arg(a, "wav_samp_rate", 16000))
            padding_type = _get_arg(a, "padding_type", "repeat")
            random_start = bool(_get_arg(a, "random_start", False))

            def _collate(batch):
                return variable_multi_view_collate_fn(
                    batch,
                    top_k=top_k,
                    min_duration=min_duration,
                    max_duration=max_duration,
                    sample_rate=wav_sr,
                    padding_type=padding_type,
                    random_start=random_start,
                )
        else:
            views = _get_arg(a, "views", [1, 2, 3, 4])
            wav_sr = int(_get_arg(a, "wav_samp_rate", 16000))
            padding_type = _get_arg(a, "padding_type", "repeat")
            random_start = bool(_get_arg(a, "random_start", False))
            view_padding_configs = _get_arg(a, "view_padding_configs", None)

            def _collate(batch):
                return multi_view_collate_fn(
                    batch,
                    views=views,
                    sample_rate=wav_sr,
                    padding_type=padding_type,
                    random_start=random_start,
                    view_padding_configs=view_padding_configs,
                )
        self.collate_fn = _collate

        if self.chunking_eval:
            self.eval_collator = ChunkingCollator(
                chunk_size=int(_get_arg(a, "chunk_size", 16000)),
                overlap_size=int(_get_arg(a, "overlap_size", 8000)),
                enable_chunking=True,
            )
        else:
            self.eval_collator = None

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
        loader = WebLoader(
            self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.eval_collator,
            persistent_workers=bool(self.hparams.num_workers),
        )
        return self._apply_epoch_batches(loader, split="eval", drop_last=False)


# Backward-compatible alias for explicit optimized naming.
NormalDataModuleMV = NormalDataModule
