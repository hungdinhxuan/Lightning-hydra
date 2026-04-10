"""Multi-band consistency training (MBCT) Lightning module.

Batches are dicts mapping band names (e.g. ``normal``, ``narrowband``, ``wideband``) to
``(waveform, label)`` pairs, as produced by :func:`src.data.components.collate_fn.mbct_collate_fn`.

Training and validation logic matches :class:`MDTLitModule`. Configure ``weighted_views`` in
Hydra with the **same keys** as ``band_configs`` in the datamodule (default bands below).

Typical use: :class:`src.data.normal_mbct_datamodule.NormalMBCTDataModule` (fix duration in
dataset) + this module + ``weighted_views`` for each band.
"""
from __future__ import annotations

from typing import Dict, FrozenSet

from src.models.base.mdt_module import MDTLitModule

# Default keys when mbct_collate_fn uses built-in band_configs (see collate_fn.py).
DEFAULT_MBCT_BAND_KEYS: FrozenSet[str] = frozenset(
    ("normal", "narrowband", "wideband")
)

# Default composite keys for :class:`src.data.normal_mbct_mdt_datamodule.NormalMBCTMDTDataModule`
# (views 1–4 seconds × three bands). Match ``weighted_views`` in Hydra when using MDT+MBCT.
DEFAULT_MBCT_MDT_COMPOSITE_KEYS: FrozenSet[str] = frozenset(
    f"{v}_{b}"
    for v in ("1", "2", "3", "4")
    for b in ("normal", "narrowband", "wideband")
)


class MBCTLitModule(MDTLitModule):
    """Same training loop as MDT, with string band keys instead of duration views ``\"1\"``…``\"4\"``."""

    @staticmethod
    def band_keys_match_weighted_views(
        weighted_views: Dict[str, float], batch: Dict[str, object]
    ) -> bool:
        """Return True if every key in ``weighted_views`` exists in ``batch`` (sanity check)."""
        return set(weighted_views.keys()).issubset(set(batch.keys()))
