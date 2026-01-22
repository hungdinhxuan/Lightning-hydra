#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline low-pass filter augmentation using torch-audiomentations.

Usage (example):

    python scripts/augment_low_pass_torch_audiomentations.py \\
        --input_dir /path/to/input_wavs \\
        --output_dir /path/to/output_wavs \\
        --sample_rate 16000 \\
        --min_cutoff 2000 \\
        --max_cutoff 7500 \\
        --device cuda

This script:
  - scans an input folder for audio files (*.wav by default)
  - loads each file (resampled to `--sample_rate`)
  - applies `torch_audiomentations.LowPassFilter` on GPU/CPU
  - saves the augmented audio to the output folder (mirroring directory structure)
"""

import argparse
import os
import sys
import glob
from typing import List

import numpy as np
import torch

# Reuse project audio IO helpers
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.data.components.dataio import load_audio, save_audio  # noqa: E402

try:
    from torch_audiomentations import LowPassFilter  # noqa: E402
except ImportError as e:  # pragma: no cover - runtime check
    raise ImportError(
        "torch-audiomentations is required for this script.\n"
        "Install it with: pip install torch-audiomentations"
    ) from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply torch-audiomentations LowPassFilter to all audio files in a folder and save augmented audio."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing audio files (will be scanned recursively).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save augmented audio (directory structure is mirrored).",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target sample rate for loading/saving audio (default: 16000).",
    )
    parser.add_argument(
        "--target_rate",
        type=int,
        default=16000,
        help="Target sample rate for saving audio (default: 16000).",
    )
    parser.add_argument(
        "--min_cutoff",
        type=float,
        default=2000.0,
        help="Minimum cutoff frequency (Hz) for LowPassFilter (default: 2000).",
    )
    parser.add_argument(
        "--max_cutoff",
        type=float,
        default=7500.0,
        help="Maximum cutoff frequency (Hz) for LowPassFilter (default: 7500).",
    )
    parser.add_argument(
        "--prob",
        type=float,
        default=1.0,
        help="Probability p of applying the augmentation (default: 1.0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run augmentation on (default: cuda, falls back to CPU if not available).",
    )
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="*",
        default=["*.wav"],
        help="Glob patterns of audio files to process (default: *.wav).",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Optional limit on number of files to process (0 means all).",
    )
    return parser.parse_args()


def collect_audio_files(input_dir: str, patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        glob_pattern = os.path.join(input_dir, "**", pattern)
        files.extend(glob.glob(glob_pattern, recursive=True))
    files = sorted(list({os.path.abspath(f) for f in files}))
    return files


def main() -> None:
    args = parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Determine device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # Prepare augmentation
    aug = LowPassFilter(
        min_cutoff_freq=args.min_cutoff,
        max_cutoff_freq=args.max_cutoff,
        p=args.prob,
        sample_rate=args.sample_rate,
        output_type="dict",
        #target_rate=args.target_rate,
    ).to(device)

    # Collect files
    audio_files = collect_audio_files(input_dir, args.patterns)
    if args.max_files > 0:
        audio_files = audio_files[: args.max_files]

    if not audio_files:
        print(f"[WARN] No audio files found in {input_dir} with patterns {args.patterns}")
        return

    print(f"[INFO] Found {len(audio_files)} files. Starting augmentation...")

    for idx, in_path in enumerate(audio_files, start=1):
        rel_path = os.path.relpath(in_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            # Load audio as numpy (mono), resampled to target sample_rate
            audio_np = load_audio(in_path, sr=args.sample_rate, cache_dir=None)

            if audio_np.ndim > 1:
                # Flatten to mono if needed
                audio_np = np.mean(audio_np, axis=0)

            # Convert to torch tensor on device: (1, 1, samples)
            audio_tensor = torch.from_numpy(audio_np.astype(np.float32))
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)
            elif audio_tensor.ndim == 2:
                audio_tensor = audio_tensor.unsqueeze(0)
            audio_tensor = audio_tensor.to(device)

            # Apply low-pass filter
            with torch.no_grad():
                out = aug(samples=audio_tensor, sample_rate=args.sample_rate)

            if isinstance(out, dict):
                out_tensor = out["samples"]
            elif isinstance(out, torch.Tensor):
                out_tensor = out
            else:
                out_tensor = out.samples

            # Remove batch/channel dims -> (samples,)
            if out_tensor.ndim == 3:
                out_tensor = out_tensor[0, 0]
            elif out_tensor.ndim == 2:
                out_tensor = out_tensor[0]

            out_np = out_tensor.detach().cpu().numpy()

            # Save augmented audio
            save_audio(out_path, out_np, sr=args.sample_rate)

            if idx % 50 == 0 or idx == len(audio_files):
                print(f"[INFO] Processed {idx}/{len(audio_files)} files")

        except Exception as exc:  # pragma: no cover - diagnostic
            print(f"[ERROR] Failed to process {in_path}: {exc}")

    print(f"[DONE] Augmented {len(audio_files)} files.")
    print(f"[OUTPUT] Files saved under: {output_dir}")


if __name__ == "__main__":
    main()


