#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader, Sampler

from src.data.replay_multiview_datamodule import ReplayDataModule, ReplayDataset


class ReplayBalanceClassSampler(Sampler):
    """Replay sampler that mirrors novel class mix inside replay part of each batch."""

    def __init__(self, dataset: ReplayDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.novel_ratio = dataset.novel_ratio
        self.replay_ratio = dataset.replay_ratio

        self.novel_per_batch = int(batch_size * self.novel_ratio)
        self.replay_per_batch = int(batch_size * self.replay_ratio)
        total_per_batch = self.novel_per_batch + self.replay_per_batch
        if total_per_batch > batch_size:
            self.novel_per_batch = batch_size - self.replay_per_batch

        if self.novel_per_batch <= 0:
            raise ValueError("novel_ratio and batch_size must yield at least one novel sample per batch")
        if self.replay_per_batch <= 0:
            raise ValueError("replay_ratio and batch_size must yield at least one replay sample per batch")

        self.total_per_batch = self.novel_per_batch + self.replay_per_batch
        self.novel_label_to_indices = self._build_label_to_indices(
            dataset.novel_list_IDs, dataset.novel_labels, split_name="novel"
        )
        self.replay_label_to_indices = self._build_label_to_indices(
            dataset.replay_list_IDs, dataset.replay_labels, split_name="replay"
        )
        self._replay_cursors: Dict[int, int] = defaultdict(int)

        print(
            "Class-balanced batch composition: "
            f"{self.novel_per_batch} novel + {self.replay_per_batch} replay = {self.total_per_batch} total"
        )

    @staticmethod
    def _build_label_to_indices(files: List[str], labels: Dict[str, int], split_name: str) -> Dict[int, List[int]]:
        label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, utt_id in enumerate(files):
            if utt_id not in labels:
                raise KeyError(f"{split_name} label missing for {utt_id}")
            label_to_indices[int(labels[utt_id])].append(idx)
        return dict(label_to_indices)

    def _draw_replay_indices(self, label: int, count: int) -> List[int]:
        if count <= 0:
            return []
        pool = self.replay_label_to_indices.get(label, [])
        if not pool:
            raise ValueError(f"Replay set has no samples for label {label}")

        selected: List[int] = []
        while len(selected) < count:
            cursor = self._replay_cursors[label]
            if cursor == 0 and self.shuffle:
                random.shuffle(pool)
            take = min(count - len(selected), len(pool) - cursor)
            selected.extend(pool[cursor : cursor + take])
            cursor = (cursor + take) % len(pool)
            self._replay_cursors[label] = cursor
        return selected

    def _replay_counts_for_novel_batch(self, novel_batch: List[int]) -> Dict[int, int]:
        novel_counts: Dict[int, int] = defaultdict(int)
        for idx in novel_batch:
            utt_id = self.dataset.novel_list_IDs[idx]
            novel_counts[int(self.dataset.novel_labels[utt_id])] += 1

        if len(novel_batch) == self.replay_per_batch:
            return dict(novel_counts)

        labels = sorted(novel_counts)
        raw_counts = {
            label: (novel_counts[label] / len(novel_batch)) * self.replay_per_batch
            for label in labels
        }
        replay_counts = {label: int(raw_counts[label]) for label in labels}
        remaining = self.replay_per_batch - sum(replay_counts.values())
        labels_by_remainder = sorted(
            labels,
            key=lambda label: (raw_counts[label] - replay_counts[label], novel_counts[label]),
            reverse=True,
        )
        for label in labels_by_remainder[:remaining]:
            replay_counts[label] += 1
        return replay_counts

    def __iter__(self):
        novel_indices = list(range(len(self.dataset.novel_list_IDs)))
        if self.shuffle:
            random.shuffle(novel_indices)

        for i in range(0, len(novel_indices), self.novel_per_batch):
            novel_batch = novel_indices[i : i + self.novel_per_batch]
            replay_counts = self._replay_counts_for_novel_batch(novel_batch)

            batch_indices = [("novel", idx) for idx in novel_batch]
            for label, count in replay_counts.items():
                batch_indices.extend(("replay", idx) for idx in self._draw_replay_indices(label, count))

            if self.shuffle:
                random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return math.ceil(len(self.dataset.novel_list_IDs) / self.novel_per_batch)


class ReplayBalanceClassDataLoader(DataLoader):
    """DataLoader using class-balanced replay batches."""

    def __init__(
        self,
        dataset: ReplayDataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
        collate_fn=None,
        persistent_workers: bool = False,
        **kwargs,
    ):
        sampler = ReplayBalanceClassSampler(dataset, batch_size, shuffle)
        super().__init__(
            dataset=dataset,
            batch_size=None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn or self._default_collate_fn,
            persistent_workers=persistent_workers,
            **kwargs,
        )

    def _default_collate_fn(self, batch):
        if len(batch) == 1 and isinstance(batch[0], list):
            batch_data = batch[0]
        else:
            batch_data = batch
        if batch_data and len(batch_data[0]) == 3:
            samples, targets, sources = zip(*batch_data)
            return list(samples), list(targets), list(sources)
        samples, targets = zip(*batch_data)
        return list(samples), list(targets)


class ReplayBalanceClassDataModule(ReplayDataModule):
    """ReplayDataModule variant where replay class counts mirror novel batch counts."""

    def train_dataloader(self) -> DataLoader[Any]:
        return ReplayBalanceClassDataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )
