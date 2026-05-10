from pathlib import Path

import hydra
import pytest
import torch
import torchaudio
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from lightning.pytorch import Trainer
from omegaconf import open_dict

from src.data.feature_datamodule import FeatureProtocolDataModule, ProtocolFeatureDataset
from src.models.components.lcnn import LCNN


def _write_tone(path: Path, freq: float, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = torch.linspace(0, 0.35, int(sample_rate * 0.35))
    waveform = 0.1 * torch.sin(2 * torch.pi * freq * t)
    torchaudio.save(str(path), waveform.unsqueeze(0), sample_rate)


@pytest.fixture()
def lcnn_feature_root(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
        ("bonafide_train.wav", "train", "bonafide", 220.0),
        ("spoof_train.wav", "train", "spoof", 330.0),
        ("bonafide_dev.wav", "dev", "bonafide", 440.0),
        ("spoof_dev.wav", "dev", "spoof", 550.0),
        ("bonafide_eval.wav", "eval", "bonafide", 660.0),
        ("spoof_eval.wav", "eval", "spoof", 770.0),
    ]
    protocol_lines = []
    for name, subset, label, freq in rows:
        _write_tone(tmp_path / name, freq)
        protocol_lines.append(f"{name} {subset} {label}\n")

    protocol_path = tmp_path / "protocol.txt"
    protocol_path.write_text("".join(protocol_lines), encoding="utf-8")
    return tmp_path, protocol_path


def test_lcnn_forward_feature_tensor() -> None:
    model = LCNN(num_coefficients=384, output_dim=2, dropout=0.0)
    x = torch.randn(2, 1, 384, 32)
    logits = model(x)
    assert logits.shape == (2, 2)


def test_feature_protocol_dataset_shapes(lcnn_feature_root: tuple[Path, Path]) -> None:
    data_dir, protocol_path = lcnn_feature_root
    dataset = ProtocolFeatureDataset(
        data_dir=str(data_dir),
        protocol_path=str(protocol_path),
        subset="train",
        feature_type="lfcc",
        max_frames=32,
    )
    x, y = dataset[0]
    assert x.shape == (1, 384, 32)
    assert y.dtype == torch.long


def test_feature_datamodule_batches(lcnn_feature_root: tuple[Path, Path]) -> None:
    data_dir, protocol_path = lcnn_feature_root
    dm = FeatureProtocolDataModule(
        data_dir=str(data_dir),
        protocol_path=str(protocol_path),
        batch_size=2,
        num_workers=0,
        args={"feature_type": "mfcc", "max_frames": 32},
    )
    dm.setup()
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (2, 1, 384, 32)
    assert y.shape == (2,)


def test_lcnn_hydra_smoke_train(lcnn_feature_root: tuple[Path, Path], tmp_path: Path) -> None:
    data_dir, protocol_path = lcnn_feature_root
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                "experiment=lcnn_feature_smoke",
                f"data.data_dir={data_dir}",
                f"data.protocol_path={protocol_path}",
                f"paths.output_dir={tmp_path}",
                f"paths.log_dir={tmp_path}",
                "extras.print_config=false",
                "extras.enforce_tags=false",
            ],
            return_hydra_config=True,
        )

    with open_dict(cfg):
        cfg.trainer.enable_checkpointing = False
        cfg.trainer.logger = False
        cfg.trainer.enable_model_summary = False

    datamodule = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate(cfg.model)
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.fit(model=model, datamodule=datamodule)

