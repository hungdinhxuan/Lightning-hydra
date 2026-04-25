"""Tests for mbct_collate_fn (multi-band consistency training collate)."""

import os
import sys

import numpy as np
import pytest
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

pytest.importorskip("torchaudio")

from src.data.components.collate_fn import mbct_collate_fn, mbct_mdt_collate_fn  # noqa: E402
from src.models.base.mbct_module import DEFAULT_MBCT_MDT_COMPOSITE_KEYS  # noqa: E402


def test_mbct_collate_fn_default_keys_shapes_and_labels():
    """Default band configs yield three views; each view is (B, T) waveforms + long labels."""
    sr = 16000
    t = 3200
    batch = [
        (torch.randn(t, dtype=torch.float32), 0),
        (torch.randn(t, dtype=torch.float32), 1),
    ]

    out = mbct_collate_fn(batch, sample_rate=sr)

    assert set(out.keys()) == {"normal", "narrowband", "wideband"}
    for name, (seq, labels) in out.items():
        assert seq.ndim == 2
        assert seq.shape[0] == 2
        assert seq.dtype == torch.float32
        assert labels.shape == (2,)
        assert labels.dtype == torch.long
        assert torch.equal(labels, torch.tensor([0, 1]))


def test_mbct_collate_fn_all_bands_same_length():
    """Resample and low-pass keep length T; stacked batch shapes match across bands."""
    sr = 16000
    t = 4000
    batch = [(torch.randn(t), 0), (torch.randn(t), 1)]

    out = mbct_collate_fn(batch, sample_rate=sr)

    shapes = {name: out[name][0].shape for name in out}
    assert len(set(shapes.values())) == 1


def test_mbct_collate_fn_normal_is_identity_numpy_input():
    """Normal band leaves audio unchanged; numpy samples are accepted."""
    sr = 16000
    x_np = np.random.randn(1600).astype(np.float32)
    batch = [(x_np, 3)]

    out = mbct_collate_fn(
        batch,
        sample_rate=sr,
        band_configs={"normal": {"type": "normal"}},
    )

    seq, labels = out["normal"]
    assert seq.shape == (1, 1600)
    assert torch.allclose(seq[0], torch.from_numpy(x_np))
    assert labels.item() == 3


def test_mbct_max_length_sec_fixed_duration():
    """max_length_sec crops/pads to a fixed number of samples before band transforms."""
    sr = 16000
    long_w = torch.randn(50_000)
    batch = [(long_w, 0)]

    max_sec = 0.25
    target_len = int(max_sec * sr)
    out = mbct_collate_fn(
        batch,
        sample_rate=sr,
        max_length_sec=max_sec,
        padding_type="repeat",
        random_start=False,
    )

    for name, (seq, _) in out.items():
        assert seq.shape == (1, target_len)


def test_mbct_unknown_band_type_raises():
    batch = [(torch.randn(100), 0)]

    with pytest.raises(ValueError, match="Unknown band config type"):
        mbct_collate_fn(
            batch,
            band_configs={"weird": {"type": "not_a_real_band"}},
        )


def test_mbct_narrowband_differs_from_normal():
    """Narrowband path (resample) should not match identity normal for generic noise."""
    sr = 16000
    t = 8000
    rng = torch.Generator().manual_seed(42)
    x = torch.randn(t, generator=rng)
    batch = [(x, 0)]

    out = mbct_collate_fn(batch, sample_rate=sr)
    normal = out["normal"][0][0]
    narrow = out["narrowband"][0][0]

    assert normal.shape == narrow.shape
    assert not torch.allclose(normal, narrow, atol=1e-6)


def test_mbct_custom_wideband_cutoff_runs():
    """Custom cutoff in config is forwarded without error."""
    sr = 16000
    batch = [(torch.randn(4000), 1)]

    out = mbct_collate_fn(
        batch,
        sample_rate=sr,
        band_configs={
            "wide": {"type": "wideband", "cutoff_hz": 6000},
        },
    )

    seq, labels = out["wide"]
    assert seq.shape == (1, 4000)
    assert labels.item() == 1


def test_mbct_mdt_collate_fn_keys_and_lengths():
    """MDT×MBCT: composite keys view_band; each view has different T = view * sr."""
    sr = 16000
    t = 64000
    batch = [
        (torch.randn(t, dtype=torch.float32), 0),
        (torch.randn(t, dtype=torch.float32), 1),
    ]
    vpc = {
        "1": {"padding_type": "repeat", "random_start": False},
        "2": {"padding_type": "repeat", "random_start": False},
    }
    out = mbct_mdt_collate_fn(
        batch,
        sample_rate=sr,
        view_padding_configs=vpc,
        band_configs={
            "normal": {"type": "normal"},
            "narrowband": {"type": "narrowband"},
            "wideband": {"type": "wideband", "cutoff_hz": 7800},
        },
    )
    expected = {f"{v}_{b}" for v in ("1", "2") for b in ("normal", "narrowband", "wideband")}
    assert set(out.keys()) == expected
    assert out["1_normal"][0].shape == (2, 1 * sr)
    assert out["2_normal"][0].shape == (2, 2 * sr)
    assert torch.equal(out["1_normal"][1], torch.tensor([0, 1]))


def test_mbct_mdt_default_matches_module_constant():
    """Default 4-view × 3-band keys align with DEFAULT_MBCT_MDT_COMPOSITE_KEYS."""
    sr = 16000
    batch = [(torch.randn(64000), 0)]
    out = mbct_mdt_collate_fn(batch, sample_rate=sr)
    assert set(out.keys()) == set(DEFAULT_MBCT_MDT_COMPOSITE_KEYS)
