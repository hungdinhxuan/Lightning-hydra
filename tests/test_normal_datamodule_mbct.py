"""NormalMBCTDataModule: fix-duration dataset + MBCT collate."""

import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from src.data.normal_mbct_datamodule import NormalMBCTDataModule  # noqa: E402


def test_normal_mbct_datamodule_collate_returns_band_dict_fixed_length():
    """Samples are fixed length (pad in dataset); collate stacks then applies band transforms."""
    args = {
        "wav_samp_rate": 16000,
        "trim_length": 8000,
        "padding_type": "repeat",
        "random_start": False,
        "protocol_path": "dummy",
        "augmentation_methods": [],
        "mbct_fix_duration_in_dataset": True,
    }
    dm = NormalMBCTDataModule(data_dir=".", batch_size=2, num_workers=0, args=args)
    # Fixed-length waveforms as after Dataset_for pad
    batch = [
        (torch.randn(8000, dtype=torch.float32), 0),
        (torch.randn(8000, dtype=torch.float32), 1),
    ]
    out = dm.collate_fn(batch)
    assert set(out.keys()) == {"normal", "narrowband", "wideband"}
    for _name, (seq, labels) in out.items():
        assert seq.shape == (2, 8000)
        assert labels.tolist() == [0, 1]
