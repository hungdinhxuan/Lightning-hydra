from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from src.utils import load_ln_model_weights


CheckpointFormat = Literal["auto", "lightning", "raw"]


def load_frozen_teacher_model(
    model_factory: Callable[[], nn.Module],
    checkpoint_path: str,
    checkpoint_format: CheckpointFormat = "auto",
    strict: bool = True,
) -> nn.Module:
    """Build teacher model, load checkpoint, freeze, and switch to eval mode."""

    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"teacher checkpoint not found: {path}")

    model = model_factory()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(checkpoint)

    if checkpoint_format == "auto":
        checkpoint_format = "lightning" if _looks_like_lightning_state_dict(state_dict) else "raw"

    if checkpoint_format == "lightning":
        model = load_ln_model_weights(model, state_dict)
    elif checkpoint_format == "raw":
        model.load_state_dict(_normalize_raw_keys(state_dict), strict=strict)
    else:
        raise ValueError(f"unsupported checkpoint_format: {checkpoint_format!r}")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError("teacher checkpoint must be a state_dict or contain a 'state_dict' key")
    return {
        str(key): value
        for key, value in checkpoint.items()
        if torch.is_tensor(value)
    }


def _looks_like_lightning_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("net.") for key in state_dict)


def _normalize_raw_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.replace("module.", "").replace("_orig_mod.", ""): value
        for key, value in state_dict.items()
    }
