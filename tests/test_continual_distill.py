from pathlib import Path
import os
import sys

import pytest
import torch
from torch import nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from src.data.components.collate_fn import multi_view_collate_fn
from src.models.components.continual_distill_strategy import (
    SOURCE_NOVEL,
    SOURCE_REPLAY,
    ContinualDistillStrategy,
)
from src.models.components.distillation import LogitKDLoss
from src.models.components.teacher_wrapper import load_frozen_teacher_model


def test_logit_kd_loss_is_finite_and_zero_for_empty_mask() -> None:
    loss_fn = LogitKDLoss(temperature=2.0)
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    teacher = torch.tensor([[0.5, 0.5], [0.2, 0.8]])

    loss = loss_fn(student, teacher)
    assert torch.isfinite(loss)
    assert loss.item() >= 0

    empty_loss = loss_fn(student[:0], teacher[:0])
    assert empty_loss.item() == 0.0


def test_replay_only_strategy_builds_replay_mask() -> None:
    strategy = ContinualDistillStrategy(apply_on="replay_only")
    sources = torch.tensor([SOURCE_NOVEL, SOURCE_REPLAY, SOURCE_REPLAY, SOURCE_NOVEL])

    mask = strategy.build_mask(sources, batch_size=4, device=torch.device("cpu"))

    assert mask.tolist() == [False, True, True, False]


def test_replay_only_strategy_requires_source_metadata() -> None:
    strategy = ContinualDistillStrategy(apply_on="replay_only")

    with pytest.raises(ValueError, match="data.return_source=true"):
        strategy.build_mask(None, batch_size=2, device=torch.device("cpu"))


def test_multi_view_collate_preserves_source_metadata() -> None:
    batch = [
        (torch.ones(4), 1, SOURCE_NOVEL),
        (torch.zeros(6), 0, SOURCE_REPLAY),
    ]

    out = multi_view_collate_fn(
        batch,
        sample_rate=4,
        view_padding_configs={
            "1": {"padding_type": "repeat", "random_start": False},
            "2": {"padding_type": "repeat", "random_start": False},
        },
    )

    assert set(out.keys()) == {1, 2}
    x_view, labels, sources = out[1]
    assert x_view.shape == (2, 4)
    assert labels.tolist() == [1, 0]
    assert sources.tolist() == [SOURCE_NOVEL, SOURCE_REPLAY]


def test_load_frozen_teacher_model_from_raw_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "teacher.pt"
    model = nn.Linear(3, 2)
    torch.save(model.state_dict(), checkpoint_path)

    teacher = load_frozen_teacher_model(
        model_factory=lambda: nn.Linear(3, 2),
        checkpoint_path=str(checkpoint_path),
        checkpoint_format="raw",
    )

    assert not teacher.training
    assert all(not param.requires_grad for param in teacher.parameters())
