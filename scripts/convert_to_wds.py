#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert protocol-based wav dataset to WebDataset shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

# Ensure `src` is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import webdataset as wds
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'webdataset'. Install it with `uv sync` "
        "or `pip install webdataset` and re-run this script."
    ) from exc

from src.data.dataset_optimized import read_protocol, split_entries_by_subset


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _write_subset(
    subset: str,
    entries,
    input_wav_dir: Path,
    output_dir: Path,
    shard_size_bytes: int,
) -> Dict[str, int]:
    pattern = output_dir / f"{subset}-%04d.tar"
    _ensure_parent(pattern)
    n_samples = 0
    skipped = 0

    with wds.ShardWriter(str(pattern), maxsize=shard_size_bytes) as sink:
        for e in entries:
            wav_path = input_wav_dir / e.relpath
            if not wav_path.exists():
                skipped += 1
                continue

            key = e.key
            sample = {
                "__key__": key,
                "wav": _load_bytes(wav_path),
                "label": e.label_str,
                "json": json.dumps(
                    {
                        "relpath": e.relpath,
                        "subset": subset,
                        "label": e.label_str,
                    }
                ).encode("utf-8"),
            }
            sink.write(sample)
            n_samples += 1

    return {"written": n_samples, "skipped": skipped}


def build_webdataset(
    input_wav_dir: str,
    protocol_path: str,
    output_dir: str,
    shard_size_mb: int = 256,
) -> Dict[str, Dict[str, int]]:
    input_root = Path(input_wav_dir)
    protocol = Path(protocol_path)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    entries = read_protocol(str(protocol))
    by_subset = split_entries_by_subset(entries)
    maxsize = shard_size_mb * 1024 * 1024

    summary: Dict[str, Dict[str, int]] = {}
    for subset in ("train", "dev", "eval"):
        summary[subset] = _write_subset(
            subset=subset,
            entries=by_subset.get(subset, []),
            input_wav_dir=input_root,
            output_dir=out_root,
            shard_size_bytes=maxsize,
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert wav+protocol dataset to WebDataset shards.")
    parser.add_argument("--input_wav_dir", required=True, help="Directory containing wav files")
    parser.add_argument("--protocol_path", required=True, help="Path to protocol.txt")
    parser.add_argument("--output_dir", required=True, help="Output directory for train/dev/eval shards")
    parser.add_argument("--shard_size_mb", type=int, default=256, help="Shard size target in MB (100-500 recommended)")
    parser.add_argument(
        "--use_dev_shm",
        action="store_true",
        help="Write shards under /dev/shm/<basename(output_dir)> for faster IO when available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if args.use_dev_shm and os.path.isdir("/dev/shm"):
        output_dir = str(Path("/dev/shm") / Path(output_dir).name)

    summary = build_webdataset(
        input_wav_dir=args.input_wav_dir,
        protocol_path=args.protocol_path,
        output_dir=output_dir,
        shard_size_mb=args.shard_size_mb,
    )
    print(f"WebDataset shards written to: {output_dir}")
    for subset, stats in summary.items():
        print(f"{subset}: written={stats['written']} skipped={stats['skipped']}")


if __name__ == "__main__":
    main()
