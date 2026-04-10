#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WebDataset utilities for optimized audio loading."""

from __future__ import annotations

import os
import shlex
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torchaudio
import webdataset as wds
from torch.utils.data import default_collate
from webdataset import WebLoader

from src.data.components.dataio import pad_tensor
from src.data.components.augwrapper import SUPPORTED_AUGMENTATION

for _aug in SUPPORTED_AUGMENTATION:
    exec(f"from src.data.components.augwrapper import {_aug}")


LABEL_TO_INT = {"bonafide": 1, "spoof": 0}
VALID_SUBSETS = {"train", "dev", "eval", "test"}


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


@dataclass
class ProtocolEntry:
    relpath: str
    subset: str
    label_str: str

    @property
    def label_id(self) -> int:
        return LABEL_TO_INT[self.label_str]

    @property
    def key(self) -> str:
        return Path(self.relpath).with_suffix("").as_posix().replace("/", "_")


def read_protocol(protocol_path: str) -> List[ProtocolEntry]:
    entries: List[ProtocolEntry] = []
    with open(protocol_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = shlex.split(raw)
            if len(parts) < 3:
                continue
            relpath, subset, label = parts[0], parts[1].lower(), parts[2].lower()
            if subset not in VALID_SUBSETS:
                continue
            if label not in LABEL_TO_INT:
                continue
            entries.append(ProtocolEntry(relpath=relpath, subset=subset, label_str=label))
    return entries


def split_entries_by_subset(entries: Sequence[ProtocolEntry]) -> Dict[str, List[ProtocolEntry]]:
    by_subset: Dict[str, List[ProtocolEntry]] = {"train": [], "dev": [], "eval": []}
    for e in entries:
        if e.subset == "test":
            by_subset["eval"].append(e)
        elif e.subset in by_subset:
            by_subset[e.subset].append(e)
    return by_subset


def _ensure_wave_1d(wav: torch.Tensor) -> torch.Tensor:
    if wav.ndim == 2:
        wav = wav[0]
    return wav.flatten()


def _decode_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    wav, sr = sample["wav"]
    wav = _ensure_wave_1d(wav).to(torch.float32)
    sample["wav"] = wav
    sample["sample_rate"] = int(sr)
    return sample


def _decode_sample_from_bytes(sample: Dict[str, Any]) -> Dict[str, Any]:
    payload = sample["wav"]
    if isinstance(payload, (bytes, bytearray)):
        wav, sr = torchaudio.load(io.BytesIO(payload))
        sample["wav"] = _ensure_wave_1d(wav).to(torch.float32)
        sample["sample_rate"] = int(sr)
        return sample
    if isinstance(payload, tuple) and len(payload) == 2:
        return _decode_sample(sample)
    raise TypeError(f"Unsupported wav payload type: {type(payload)}")


def _normalize_sr(sample: Dict[str, Any], target_sr: int) -> Dict[str, Any]:
    current_sr = int(sample.get("sample_rate", target_sr))
    if current_sr != target_sr:
        sample["wav"] = torchaudio.functional.resample(sample["wav"], current_sr, target_sr)
        sample["sample_rate"] = target_sr
    return sample


def _fix_duration(sample: Dict[str, Any], max_len: int, padding_type: str, random_start: bool) -> Dict[str, Any]:
    sample["wav"] = pad_tensor(sample["wav"], padding_type=padding_type, max_len=max_len, random_start=random_start)
    return sample


def _map_label(sample: Dict[str, Any]) -> Dict[str, Any]:
    label = sample["label"]
    if isinstance(label, bytes):
        label = label.decode("utf-8")
    if not isinstance(label, str):
        label = str(label)
    sample["label"] = LABEL_TO_INT[label.strip().lower()]
    return sample


def _to_tuple_train(sample: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
    return sample["wav"], int(sample["label"])


def _to_tuple_eval(sample: Dict[str, Any], return_key: bool) -> Tuple[torch.Tensor, Any]:
    if return_key:
        key = sample.get("__key__", "")
        return sample["wav"], key
    return sample["wav"], int(sample["label"])


def _build_key_to_relpath(protocol_path: str, subset: str) -> Dict[str, str]:
    """Build a mapping from WebDataset sample key to original relative file path."""
    entries = read_protocol(protocol_path)
    mapping: Dict[str, str] = {}
    for e in entries:
        norm_subset = "eval" if e.subset == "test" else e.subset
        if norm_subset == subset:
            mapping[e.key] = e.relpath
    return mapping


def _make_augment_map(
    augmentation_methods: List[str],
    args: Any,
    sample_rate: int,
    data_dir: Optional[str] = None,
    key_to_relpath: Optional[Dict[str, str]] = None,
):
    """Create a per-sample augmentation map function for the WebDataset pipeline.

    Follows the same approach as Dataset_for in normal_datamodule.py:
    randomly pick one augmentation method per sample and apply it.
    """
    aug_count = len(augmentation_methods)
    _globals = globals()

    def _augment(sample: Dict[str, Any]) -> Dict[str, Any]:
        aug_idx = random.randrange(aug_count)
        method_name = augmentation_methods[aug_idx]

        wav = sample["wav"]
        sr = sample.get("sample_rate", sample_rate)

        x_np = wav.numpy() if isinstance(wav, torch.Tensor) else wav

        audio_path = None
        if key_to_relpath is not None and data_dir is not None:
            key = sample.get("__key__", "")
            relpath = key_to_relpath.get(key)
            if relpath is not None:
                audio_path = os.path.join(data_dir, relpath)

        x_aug = _globals[method_name](x_np, args, sr, audio_path=audio_path)

        if isinstance(x_aug, torch.Tensor):
            sample["wav"] = x_aug.to(torch.float32)
        else:
            sample["wav"] = torch.as_tensor(x_aug, dtype=torch.float32).clone()

        return sample

    return _augment


def _build_shard_pattern(root_dir: str, subset: str, use_braces: bool = True) -> str:
    root = Path(root_dir)
    shards = sorted(root.glob(f"{subset}-*.tar"))
    if not shards:
        raise FileNotFoundError(f"No shards found for subset '{subset}' in {root_dir}")

    if not use_braces:
        return str(root / f"{subset}-*.tar")

    ids: List[int] = []
    for p in shards:
        stem = p.stem
        suffix = stem.split("-")[-1]
        if suffix.isdigit():
            ids.append(int(suffix))
    if not ids:
        return str(root / f"{subset}-*.tar")

    width = max(len(str(max(ids))), 4)
    return str(root / f"{subset}-{{{min(ids):0{width}d}..{max(ids):0{width}d}}}.tar")


def get_wds_dataset(
    shards_root: str,
    subset: str,
    *,
    sample_rate: int = 16000,
    trim_length: Optional[int] = None,
    padding_type: str = "repeat",
    random_start: bool = False,
    shard_shuffle: int = 100,
    sample_shuffle: int = 2000,
    include_eval_key: bool = False,
    use_brace_pattern: bool = True,
    nodesplitter: Optional[Any] = wds.split_by_node,
    augmentation_methods: Optional[List[str]] = None,
    augmentation_args: Optional[Any] = None,
    augmentation_data_dir: Optional[str] = None,
    augmentation_key_map: Optional[Dict[str, str]] = None,
) -> wds.WebDataset:
    urls = _build_shard_pattern(shards_root, subset=subset, use_braces=use_brace_pattern)
    ds = wds.WebDataset(
        urls,
        shardshuffle=shard_shuffle if subset == "train" else False,
        resampled=False,
        handler=wds.warn_and_continue,
        nodesplitter=nodesplitter,
    )
    ds = ds.map(_decode_sample_from_bytes)
    ds = ds.map(lambda s: _normalize_sr(s, sample_rate))
    ds = ds.map(_map_label)

    if augmentation_methods and len(augmentation_methods) > 0:
        augment_fn = _make_augment_map(
            augmentation_methods,
            augmentation_args,
            sample_rate,
            data_dir=augmentation_data_dir,
            key_to_relpath=augmentation_key_map,
        )
        ds = ds.map(augment_fn)

    if trim_length and trim_length > 0:
        ds = ds.map(lambda s: _fix_duration(s, trim_length, padding_type, random_start))

    if subset == "train" and sample_shuffle > 0:
        ds = ds.shuffle(sample_shuffle)

    if subset in {"dev", "eval"}:
        ds = ds.map(lambda s: _to_tuple_eval(s, include_eval_key))
    else:
        ds = ds.map(_to_tuple_train)
    return ds


def make_wds_collate_fn(fallback_pad: bool = True):
    def _collate(batch: Iterable[Tuple[torch.Tensor, Any]]):
        try:
            return default_collate(list(batch))
        except RuntimeError:
            if not fallback_pad:
                raise
            waves, targets = zip(*batch)
            max_len = max(w.shape[-1] for w in waves)
            padded = [pad_tensor(w, padding_type="zero", max_len=max_len, random_start=False) for w in waves]
            return torch.stack(padded, dim=0), torch.tensor(targets)

    return _collate


def build_wds_from_args(args: Any, subset: str, *, include_eval_key: bool = False) -> wds.WebDataset:
    shards_root = _get_arg(args, "wds_data_dir", _get_arg(args, "data_dir", "wds_data"))
    sample_rate = int(_get_arg(args, "wav_samp_rate", 16000))
    trim_length = _get_arg(args, "trim_length", None)
    trim_length = int(trim_length) if trim_length is not None else None
    padding_type = _get_arg(args, "padding_type", "repeat")
    random_start = bool(_get_arg(args, "random_start", False))
    shard_shuffle = int(_get_arg(args, "wds_shard_shuffle", 100))
    sample_shuffle = int(_get_arg(args, "wds_sample_shuffle", 2000))
    use_braces = bool(_get_arg(args, "wds_use_brace_pattern", True))

    shm_dir = _get_arg(args, "wds_shm_dir", None)
    if shm_dir:
        shm_candidate = Path(shm_dir)
        if shm_candidate.exists():
            shards_root = str(shm_candidate)

    # Per-sample augmentation: randomly pick one method per sample (same as
    # Dataset_for in normal_datamodule.py).
    augmentation_methods: Optional[List[str]] = None
    if subset == "train":
        methods = _get_arg(args, "augmentation_methods", [])
        if methods and len(methods) > 0:
            augmentation_methods = list(methods)
    elif subset == "dev":
        if bool(_get_arg(args, "is_dev_aug", False)):
            methods = _get_arg(args, "augmentation_methods", [])
            if methods and len(methods) > 0:
                augmentation_methods = list(methods)
    elif subset == "eval":
        eval_aug = _get_arg(args, "eval_augment", None)
        if eval_aug:
            augmentation_methods = [eval_aug]

    augmentation_key_map: Optional[Dict[str, str]] = None
    augmentation_data_dir: Optional[str] = None
    if augmentation_methods:
        protocol_path = _get_arg(args, "protocol_path", None)
        augmentation_data_dir = _get_arg(args, "data_dir", None)
        if protocol_path and os.path.exists(protocol_path):
            augmentation_key_map = _build_key_to_relpath(protocol_path, subset)
            print(
                f"[WDS] Augmentation enabled for '{subset}' split: "
                f"{augmentation_methods} "
                f"({len(augmentation_key_map)} key→path entries)"
            )

    return get_wds_dataset(
        shards_root=shards_root,
        subset=subset,
        sample_rate=sample_rate,
        trim_length=trim_length,
        padding_type=padding_type,
        random_start=random_start,
        shard_shuffle=shard_shuffle,
        sample_shuffle=sample_shuffle,
        include_eval_key=include_eval_key,
        use_brace_pattern=use_braces,
        augmentation_methods=augmentation_methods,
        augmentation_args=args,
        augmentation_data_dir=augmentation_data_dir,
        augmentation_key_map=augmentation_key_map,
    )


def get_wds_dataloader(
    *,
    shards_root: str,
    subset: str,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    sample_rate: int = 16000,
    trim_length: Optional[int] = None,
    padding_type: str = "repeat",
    random_start: bool = False,
    shard_shuffle: int = 100,
    sample_shuffle: int = 2000,
    include_eval_key: bool = False,
    use_brace_pattern: bool = True,
    collate_fn=None,
    augmentation_methods: Optional[List[str]] = None,
    augmentation_args: Optional[Any] = None,
    augmentation_data_dir: Optional[str] = None,
    augmentation_key_map: Optional[Dict[str, str]] = None,
) -> WebLoader:
    dataset = get_wds_dataset(
        shards_root=shards_root,
        subset=subset,
        sample_rate=sample_rate,
        trim_length=trim_length,
        padding_type=padding_type,
        random_start=random_start,
        shard_shuffle=shard_shuffle,
        sample_shuffle=sample_shuffle,
        include_eval_key=include_eval_key,
        use_brace_pattern=use_brace_pattern,
        augmentation_methods=augmentation_methods,
        augmentation_args=augmentation_args,
        augmentation_data_dir=augmentation_data_dir,
        augmentation_key_map=augmentation_key_map,
    )
    if collate_fn is None:
        collate_fn = make_wds_collate_fn(fallback_pad=True)
    return WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=bool(num_workers),
    )
