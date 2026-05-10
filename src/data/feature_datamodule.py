from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torchaudio
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.dataio import load_audio


LABELS = {"spoof": 0, "bonafide": 1}


class ProtocolFeatureDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        protocol_path: str,
        subset: str,
        feature_type: str = "lfcc",
        sample_rate: int = 16000,
        n_coefficients: int = 128,
        max_frames: int = 400,
        repeat_pad: bool = True,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.protocol_path = Path(protocol_path)
        self.subset = subset
        self.feature_type = feature_type.lower()
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.repeat_pad = repeat_pad

        if self.feature_type not in {"lfcc", "mfcc"}:
            raise ValueError("feature_type must be 'lfcc' or 'mfcc'")

        common_kwargs = {
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
        }
        if self.feature_type == "lfcc":
            self.feature_fn = torchaudio.transforms.LFCC(
                sample_rate=sample_rate,
                n_lfcc=n_coefficients,
                speckwargs=common_kwargs,
            )
        else:
            self.feature_fn = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_coefficients,
                melkwargs=common_kwargs,
            )
        self.delta_fn = torchaudio.transforms.ComputeDeltas(win_length=5, mode="replicate")
        self.items = self._read_protocol()

    def _read_protocol(self) -> list[tuple[str, int]]:
        wanted = {"eval", "test"} if self.subset == "test" else {self.subset}
        items: list[tuple[str, int]] = []
        with self.protocol_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                parts = shlex.split(raw_line.strip())
                if len(parts) < 3:
                    continue
                rel_path, subset, label = parts[:3]
                if subset not in wanted:
                    continue
                if label not in LABELS:
                    raise ValueError(f"Unsupported label '{label}' in {self.protocol_path}")
                items.append((rel_path, LABELS[label]))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def _resolve_audio_path(self, rel_path: str) -> str:
        path = Path(rel_path)
        if path.is_absolute():
            return str(path)
        return str(self.data_dir / path)

    def _fit_frames(self, features: torch.Tensor) -> torch.Tensor:
        frame_count = features.shape[-1]
        if frame_count == self.max_frames:
            return features
        if frame_count > self.max_frames:
            return features[..., : self.max_frames]

        pad_frames = self.max_frames - frame_count
        if self.repeat_pad and frame_count > 0:
            repeats = (self.max_frames + frame_count - 1) // frame_count
            repeat_shape = [1] * features.ndim
            repeat_shape[-1] = repeats
            return features.repeat(*repeat_shape)[..., : self.max_frames]
        return torch.nn.functional.pad(features, (0, pad_frames))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rel_path, label = self.items[index]
        audio = load_audio(self._resolve_audio_path(rel_path), sr=self.sample_rate)
        waveform = torch.as_tensor(audio, dtype=torch.float32)
        features = self.feature_fn(waveform)
        delta = self.delta_fn(features)
        double_delta = self.delta_fn(delta)
        features = torch.cat((features, delta, double_delta), dim=-2)
        features = self._fit_frames(features).unsqueeze(0)
        return features, torch.tensor(label, dtype=torch.long)


class FeatureProtocolDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        protocol_path: Optional[str] = None,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        self.args = args or {}
        self.protocol_path = protocol_path or self.args.get(
            "protocol_path", os.path.join(data_dir, "protocol.txt")
        )
        self.batch_size_per_device = batch_size
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 2

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by world size "
                    f"({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if self.data_train and self.data_val and self.data_test:
            return

        common = {
            "data_dir": self.data_dir,
            "protocol_path": self.protocol_path,
            "feature_type": self.args.get("feature_type", "lfcc"),
            "sample_rate": self.args.get("wav_samp_rate", 16000),
            "n_coefficients": self.args.get("n_coefficients", 128),
            "max_frames": self.args.get("max_frames", 400),
            "repeat_pad": self.args.get("repeat_pad", True),
            "n_fft": self.args.get("n_fft", 512),
            "win_length": self.args.get("win_length", 400),
            "hop_length": self.args.get("hop_length", 160),
        }
        self.data_train = ProtocolFeatureDataset(subset="train", **common)
        self.data_val = ProtocolFeatureDataset(subset="dev", **common)
        self.data_test = ProtocolFeatureDataset(subset="test", **common)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
