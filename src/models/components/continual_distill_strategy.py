from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


SOURCE_NOVEL = 0
SOURCE_REPLAY = 1


@dataclass(frozen=True)
class ContinualDistillStrategy:
    """Select which samples receive teacher-student KD."""

    apply_on: str = "replay_only"

    def __post_init__(self) -> None:
        valid = {"replay_only", "all"}
        if self.apply_on not in valid:
            raise ValueError(f"apply_on must be one of {sorted(valid)}, got {self.apply_on!r}")

    def build_mask(
        self,
        sources: torch.Tensor | Iterable[int | str] | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.apply_on == "all":
            return torch.ones(batch_size, dtype=torch.bool, device=device)

        if sources is None:
            raise ValueError(
                "distill.apply_on=replay_only requires batch source metadata. "
                "Set data.return_source=true for ReplayDataModule."
            )

        if torch.is_tensor(sources):
            source_tensor = sources.to(device=device)
        else:
            encoded = [self._encode_source(source) for source in sources]
            source_tensor = torch.tensor(encoded, dtype=torch.long, device=device)

        if source_tensor.numel() != batch_size:
            raise ValueError(
                f"source metadata length ({source_tensor.numel()}) does not match batch size ({batch_size})"
            )
        return source_tensor.reshape(-1) == SOURCE_REPLAY

    @staticmethod
    def _encode_source(source: int | str) -> int:
        if isinstance(source, str):
            lowered = source.lower()
            if lowered == "replay":
                return SOURCE_REPLAY
            if lowered == "novel":
                return SOURCE_NOVEL
            raise ValueError(f"unknown source label: {source!r}")
        return int(source)
