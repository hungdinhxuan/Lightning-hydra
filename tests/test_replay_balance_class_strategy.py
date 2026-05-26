import os
import sys
from collections import Counter

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from src.data.replay_balance_class_multiview_datamodule import ReplayBalanceClassDataLoader
from src.data.replay_multiview_datamodule import ReplayDataset
from src.models.components.continual_distill_strategy import SOURCE_NOVEL, SOURCE_REPLAY


class DummyReplayDataset(ReplayDataset):
    def _get_sample(self, utt_id, labels_dict, source=None):
        target = labels_dict[utt_id]
        sample = torch.tensor([float(target)])
        if self.return_source:
            return sample, target, source
        return sample, target


def _source_label_counts(batch, source):
    return Counter(label for _, label, item_source in batch if item_source == source)


def test_replay_balance_class_loader_mirrors_novel_class_counts() -> None:
    novel_ids = ["novel_spoof_1", "novel_spoof_2", "novel_spoof_3", "novel_bona_1"]
    replay_ids = ["replay_spoof_1", "replay_spoof_2", "replay_spoof_3", "replay_bona_1"]
    dataset = DummyReplayDataset(
        args={},
        novel_list_IDs=novel_ids,
        novel_labels={
            "novel_spoof_1": 0,
            "novel_spoof_2": 0,
            "novel_spoof_3": 0,
            "novel_bona_1": 1,
        },
        replay_list_IDs=replay_ids,
        replay_labels={
            "replay_spoof_1": 0,
            "replay_spoof_2": 0,
            "replay_spoof_3": 0,
            "replay_bona_1": 1,
        },
        base_dir="/tmp",
        novel_ratio=0.5,
        replay_ratio=0.5,
        return_source=True,
    )

    loader = ReplayBalanceClassDataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )
    batch = next(iter(loader))

    assert _source_label_counts(batch, SOURCE_NOVEL) == Counter({0: 3, 1: 1})
    assert _source_label_counts(batch, SOURCE_REPLAY) == Counter({0: 3, 1: 1})


def test_replay_balance_class_loader_scales_class_ratio_to_replay_size() -> None:
    novel_ids = [f"novel_spoof_{idx}" for idx in range(4)] + [f"novel_bona_{idx}" for idx in range(2)]
    replay_ids = [f"replay_spoof_{idx}" for idx in range(8)] + [f"replay_bona_{idx}" for idx in range(4)]
    dataset = DummyReplayDataset(
        args={},
        novel_list_IDs=novel_ids,
        novel_labels={utt_id: 0 if "spoof" in utt_id else 1 for utt_id in novel_ids},
        replay_list_IDs=replay_ids,
        replay_labels={utt_id: 0 if "spoof" in utt_id else 1 for utt_id in replay_ids},
        base_dir="/tmp",
        novel_ratio=0.6,
        replay_ratio=0.4,
        return_source=True,
    )

    loader = ReplayBalanceClassDataLoader(
        dataset=dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=lambda batch: batch,
    )
    batch = next(iter(loader))

    assert _source_label_counts(batch, SOURCE_NOVEL) == Counter({0: 4, 1: 2})
    assert _source_label_counts(batch, SOURCE_REPLAY) == Counter({0: 3, 1: 1})
