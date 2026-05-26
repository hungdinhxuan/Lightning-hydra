#!/usr/bin/env python3
"""Create per-dataset metadata CSVs for April 2026 benchmark datasets."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import shutil
import subprocess
import sys
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path("data/April_2026_benchmark")
DEFAULT_OUTPUT_NAME = "metadata.csv"
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aiff", ".aif"}
LANG_RE = re.compile(r"^[a-z]{2,3}(?:_[A-Z]{2})?$")
VC_HINT_RE = re.compile(r"(^|[^a-z])vc([^a-z]|$)|voice[-_ ]conversion|voice[-_ ]cloning|rvc", re.IGNORECASE)
TTS_HINT_RE = re.compile(r"(^|[^a-z])tts([^a-z]|$)|text[-_ ]to[-_ ]speech", re.IGNORECASE)


@dataclass(frozen=True)
class ProtocolRow:
    rel_path: str
    split: str
    attack_type: str


@dataclass(frozen=True)
class AudioInfo:
    codec: str = ""
    channel: str = ""
    duration: str = ""


def parse_protocol_line(line: str) -> ProtocolRow | None:
    line = line.strip()
    if not line:
        return None
    parts = shlex.split(line)
    if len(parts) < 3:
        raise ValueError(f"protocol line has fewer than 3 fields: {line}")
    return ProtocolRow(rel_path=parts[0], split=parts[-2], attack_type=parts[-1])


def iter_protocol_rows(protocol_path: Path) -> Iterable[ProtocolRow]:
    with protocol_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                row = parse_protocol_line(line)
            except ValueError as exc:
                raise ValueError(f"{protocol_path}:{line_no}: {exc}") from exc
            if row is not None:
                yield row


def _format_duration(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _wav_info(path: Path) -> AudioInfo | None:
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
    except (wave.Error, EOFError, OSError):
        return None

    codec_by_width = {
        1: "pcm_u8",
        2: "pcm_s16le",
        3: "pcm_s24le",
        4: "pcm_s32le",
    }
    duration = frames / rate if rate else None
    return AudioInfo(
        codec=codec_by_width.get(width, "pcm"),
        channel=str(channels) if channels else "",
        duration=_format_duration(duration),
    )


def _ffprobe_info(path: Path, ffprobe_bin: str | None) -> AudioInfo | None:
    if not ffprobe_bin:
        return None
    cmd = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,channels,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=20)
        streams = json.loads(result.stdout or "{}").get("streams", [])
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return None
    if not streams:
        return None
    stream = streams[0]
    duration = ""
    raw_duration = stream.get("duration")
    if raw_duration not in (None, "N/A"):
        try:
            duration = _format_duration(float(raw_duration))
        except (TypeError, ValueError):
            duration = ""
    return AudioInfo(
        codec=str(stream.get("codec_name") or ""),
        channel=str(stream.get("channels") or ""),
        duration=duration,
    )


def audio_info(path: Path, ffprobe_bin: str | None = None) -> AudioInfo:
    if not path.exists():
        return AudioInfo(codec=path.suffix.lower().lstrip("."))
    if path.suffix.lower() == ".wav":
        info = _wav_info(path)
        if info is not None:
            return info
    info = _ffprobe_info(path, ffprobe_bin)
    if info is not None:
        return info
    return AudioInfo(codec=path.suffix.lower().lstrip("."))


def _parts(rel_path: str) -> tuple[str, ...]:
    return Path(rel_path).parts


def infer_language(dataset_name: str, rel_path: str) -> str:
    parts = _parts(rel_path)
    if dataset_name.startswith("MLAAD") and len(parts) > 1 and LANG_RE.match(parts[1]):
        return parts[1]
    if dataset_name == "M-AILABS" and parts and LANG_RE.match(parts[0]):
        return parts[0]
    if "commonvoice26_de_en_ko_4000" in parts:
        filename = Path(rel_path).stem
        prefix = filename.split("_", 1)[0]
        return prefix if prefix in {"de", "en", "ko"} else ""
    return ""


def infer_attack_family(rel_path: str) -> str:
    rel_lower = rel_path.lower()
    if VC_HINT_RE.search(rel_lower):
        return "vc"
    if TTS_HINT_RE.search(rel_lower):
        return "tts"
    return "unk"


def infer_attack_type(dataset_name: str, rel_path: str, label: str) -> str:
    parts = _parts(rel_path)
    attack = label.lower()
    if attack in {"bonafide", "bona-fide", "real"}:
        return "bonafide"
    if dataset_name.startswith("MLAAD") and len(parts) > 2:
        return parts[2]
    if dataset_name == "2025_Kipot" and len(parts) > 2:
        return parts[2]
    if dataset_name == "artificialanalysis_audios" and parts:
        return parts[0]
    if dataset_name == "dsd_corpus_pool_24April2026" and len(parts) > 1:
        first = parts[0]
        if first in {"MLAAD_v7", "kling-ai", "April_Synthesizers"} and len(parts) > 2:
            return parts[2]
        if first == "2026_April_Dataset_Jiwon_collected" and len(parts) > 1:
            return parts[1]
        return first
    return ""


def infer_speaker_id(dataset_name: str, rel_path: str) -> str:
    parts = _parts(rel_path)
    if dataset_name == "M-AILABS" and len(parts) > 3:
        return parts[3]
    if dataset_name == "2025_Kipot" and len(parts) > 3:
        return parts[3]
    if dataset_name == "artificialanalysis_audios" and len(parts) > 1:
        return parts[1]
    if dataset_name == "dsd_corpus_pool_24April2026":
        if len(parts) > 2 and parts[0] == "2026_April_Dataset_Jiwon_collected":
            return parts[2]
        if len(parts) > 3 and parts[0] == "K-SASV-Bonafide-Training-May":
            return parts[3]
    return ""


def infer_source(dataset_name: str, rel_path: str) -> str:
    parts = _parts(rel_path)
    if dataset_name.startswith("MLAAD"):
        return "MLAAD"
    if dataset_name in {"M-AILABS", "2025_Kipot", "artificialanalysis_audios"}:
        return dataset_name
    if dataset_name == "dsd_corpus_pool_24April2026" and parts:
        return parts[0]
    return ""


def build_metadata_row(dataset_dir: Path, dataset_name: str, row: ProtocolRow, info: AudioInfo) -> dict[str, str]:
    return {
        "path": row.rel_path,
        "split": row.split,
        "attack_type": infer_attack_type(dataset_name, row.rel_path, row.attack_type),
        "attack_family": infer_attack_family(row.rel_path),
        "codec": info.codec,
        "channel": info.channel,
        "language": infer_language(dataset_name, row.rel_path),
        "speaker_id": infer_speaker_id(dataset_name, row.rel_path),
        "duration": info.duration,
        "source": infer_source(dataset_name, row.rel_path),
        "label": row.attack_type,
    }


def write_dataset_metadata(
    dataset_dir: Path,
    output_name: str,
    workers: int,
    skip_duration: bool,
    overwrite: bool,
) -> tuple[Path, int]:
    protocol_path = dataset_dir / "protocol.txt"
    if not protocol_path.exists():
        raise FileNotFoundError(f"missing protocol: {protocol_path}")

    output_path = dataset_dir / output_name
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"metadata exists, pass --overwrite: {output_path}")

    rows = list(iter_protocol_rows(protocol_path))
    ffprobe_bin = shutil.which("ffprobe")
    dataset_name = dataset_dir.name
    fieldnames = [
        "path",
        "split",
        "attack_type",
        "attack_family",
        "codec",
        "channel",
        "language",
        "speaker_id",
        "duration",
        "source",
        "label",
    ]

    def load_info(protocol_row: ProtocolRow) -> AudioInfo:
        if skip_duration:
            return AudioInfo(codec=Path(protocol_row.rel_path).suffix.lower().lstrip("."))
        return audio_info(dataset_dir / protocol_row.rel_path, ffprobe_bin=ffprobe_bin)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        if workers <= 1:
            for protocol_row in rows:
                writer.writerow(build_metadata_row(dataset_dir, dataset_name, protocol_row, load_info(protocol_row)))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(load_info, protocol_row): protocol_row for protocol_row in rows}
                for future in as_completed(futures):
                    protocol_row = futures[future]
                    writer.writerow(
                        build_metadata_row(dataset_dir, dataset_name, protocol_row, future.result())
                    )
    return output_path, len(rows)


def iter_dataset_dirs(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "protocol.txt").exists():
            yield child


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-duration", action="store_true", help="leave duration/channel blank; keep codec from extension")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if not args.root.exists():
        parser.error(f"root does not exist: {args.root}")

    total_rows = 0
    for dataset_dir in iter_dataset_dirs(args.root):
        output_path, count = write_dataset_metadata(
            dataset_dir=dataset_dir,
            output_name=args.output_name,
            workers=args.workers,
            skip_duration=args.skip_duration,
            overwrite=args.overwrite,
        )
        total_rows += count
        print(f"{dataset_dir.name}: wrote {count} rows -> {output_path}")
    print(f"total rows: {total_rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
