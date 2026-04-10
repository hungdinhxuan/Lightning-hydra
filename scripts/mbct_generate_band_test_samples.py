#!/usr/bin/env python3
"""
Load one random utterance from a protocol file, run mbct_collate_fn (same as tests),
and write normal / narrowband / wideband WAVs under out/.

Example:
  uv run python scripts/mbct_generate_band_test_samples.py \
    --out /home/hungdx/code/Lightning-hydra/out/mbct_band_test
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for p in (_ROOT, _SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import torchaudio
except ImportError as e:  # pragma: no cover
    raise SystemExit("torchaudio is required.") from e

from src.data.components.collate_fn import mbct_collate_fn  # noqa: E402


def _parse_protocol_line(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if not parts:
        return None
    return parts[0]


def _load_waveform_mono(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    return wav.squeeze(0).float()


def main() -> None:
    default_corpus = (
        Path("/home/hungdx/code/Lightning-hydra/data/DVC_DSD-Large-Corpus/raw")
        / "0_large-corpus_toys"
    )
    default_protocol = default_corpus / "protocol.txt"
    default_out = _ROOT / "out" / "mbct_band_test"

    p = argparse.ArgumentParser(description="MBCT band test WAV generator (normal / narrowband / wideband).")
    p.add_argument("--corpus-root", type=Path, default=default_corpus, help="Root containing protocol-relative paths.")
    p.add_argument("--protocol", type=Path, default=default_protocol, help="protocol.txt path.")
    p.add_argument("--out", type=Path, default=default_out, help="Output directory (created if missing).")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible random line pick.")
    p.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate (must match training).")
    p.add_argument(
        "--max-length-sec",
        type=float,
        default=None,
        help="If set, crop/pad to this duration before band transforms (same as mbct_collate_fn).",
    )
    p.add_argument(
        "--padding-type",
        type=str,
        default="repeat",
        choices=("repeat", "zero"),
    )
    p.add_argument("--random-start", action="store_true", help="Random crop when audio longer than target.")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    corpus_root = args.corpus_root.resolve()
    protocol_path = args.protocol.resolve()
    if not protocol_path.is_file():
        raise SystemExit(f"Protocol not found: {protocol_path}")

    rel_paths: list[str] = []
    with open(protocol_path, encoding="utf-8") as f:
        for line in f:
            rel = _parse_protocol_line(line)
            if rel is None:
                continue
            abs_path = corpus_root / rel
            if abs_path.is_file():
                rel_paths.append(rel)

    if not rel_paths:
        raise SystemExit(f"No existing files under {corpus_root} listed in {protocol_path}")

    chosen = random.choice(rel_paths)
    abs_audio = corpus_root / chosen

    x = _load_waveform_mono(abs_audio, args.sample_rate)
    label = 0

    collate_kw: dict = {
        "sample_rate": args.sample_rate,
        "padding_type": args.padding_type,
        "random_start": args.random_start,
    }
    if args.max_length_sec is not None:
        collate_kw["max_length_sec"] = args.max_length_sec

    views = mbct_collate_fn([(x, label)], **collate_kw)

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(chosen).stem
    meta = {
        "source_relpath": chosen,
        "source_abspath": str(abs_audio),
        "sample_rate": args.sample_rate,
        "collate_kw": {k: v for k, v in collate_kw.items()},
        "shapes": {name: list(views[name][0].shape) for name in views},
    }
    meta_path = out_dir / f"{stem}_mbct_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    for band_name, (seq, _) in views.items():
        w = seq[0].detach().cpu()
        out_wav = out_dir / f"{stem}_{band_name}.wav"
        torchaudio.save(str(out_wav), w.unsqueeze(0), args.sample_rate)

    print(f"Wrote {len(views)} band WAVs + meta under {out_dir}")
    print(f"Source: {abs_audio}")


if __name__ == "__main__":
    main()
