#!/usr/bin/env python3
"""Report May 2026 benchmark accuracy by source model type.

The May Seonghak spoof sets are single-class spoof datasets. This report uses
the benchmark run threshold metadata and groups each dataset score file by the
first path segment, e.g. `Chatterbox/example.wav -> Chatterbox`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hungdx_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from benchmark_py.protocol import parse_protocol_line
from benchmark_py.scores import parse_score_line

DEFAULT_DATASETS = [
    "May_08_2026_seonghak_spoof_audio",
    "May_08_2026_seonghak_spoof_video_converted",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report May benchmark accuracy by model type")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("logs/results/May_2026_benchmark"),
        help="Folder containing benchmark model result folders.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset name to include. Can be repeated. Defaults to the two May 08 Seonghak spoof sets.",
    )
    parser.add_argument(
        "--threshold",
        choices=["eer_threshold", "far_1pct_threshold"],
        default="eer_threshold",
        help="Threshold metric to use for accuracy decisions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Default: <results-root>/model_type_report_<timestamp>.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()


def load_protocol_labels(protocol_path: Path) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    if not protocol_path.exists():
        return labels
    with protocol_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parsed = parse_protocol_line(raw_line)
            if parsed is None:
                continue
            filepath, _, label = parsed
            labels[filepath] = label.lower()
    return labels


def load_summary_details(details_path: Path) -> Optional[dict]:
    if not details_path.exists():
        return None
    with details_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            payload = json.loads(raw_line)
            if payload.get("type") == "pooled":
                return payload.get("payload", {})
    return None


def threshold_info(details: dict, threshold_name: str) -> Tuple[float, str]:
    thresholds = details.get("results", {}).get("validation", {}).get("thresholds", {})
    item = thresholds.get(threshold_name, {})
    value = item.get("threshold")
    if value is None:
        raise RuntimeError(f"Missing threshold {threshold_name} in summary_results_details.jsonl")
    return float(value), str(item.get("source") or "summary_results_details.jsonl")


def dataset_stats(details: dict) -> Dict[str, dict]:
    return {
        str(item.get("dataset_name")): item
        for item in details.get("dataset_stats", [])
        if item.get("dataset_name")
    }


def resolve_path(path_text: str, cwd: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return cwd / path


def init_counter() -> Dict[str, float]:
    return {
        "file_count": 0,
        "n_bonafide": 0,
        "n_spoof": 0,
        "correct": 0,
        "errors": 0,
        "false_accepts": 0,
        "false_rejects": 0,
        "score_sum": 0.0,
        "score_min": math.inf,
        "score_max": -math.inf,
    }


def update_counter(counter: Dict[str, float], label: str, score: float, threshold: float) -> None:
    predicted = "bonafide" if score >= threshold else "spoof"
    counter["file_count"] += 1
    counter["score_sum"] += score
    counter["score_min"] = min(counter["score_min"], score)
    counter["score_max"] = max(counter["score_max"], score)
    if label == "bonafide":
        counter["n_bonafide"] += 1
    else:
        counter["n_spoof"] += 1
    if predicted == label:
        counter["correct"] += 1
    else:
        counter["errors"] += 1
        if label == "spoof" and predicted == "bonafide":
            counter["false_accepts"] += 1
        elif label == "bonafide" and predicted == "spoof":
            counter["false_rejects"] += 1


def counter_to_row(model_name: str, dataset: str, model_type: str, counter: Dict[str, float]) -> dict:
    total = int(counter["file_count"])
    correct = int(counter["correct"])
    errors = int(counter["errors"])
    return {
        "benchmark_model": model_name,
        "dataset": dataset,
        "model_type": model_type,
        "file_count": total,
        "n_bonafide": int(counter["n_bonafide"]),
        "n_spoof": int(counter["n_spoof"]),
        "correct": correct,
        "errors": errors,
        "false_accepts": int(counter["false_accepts"]),
        "false_rejects": int(counter["false_rejects"]),
        "accuracy_pct": 100.0 * correct / total if total else math.nan,
        "error_rate_pct": 100.0 * errors / total if total else math.nan,
        "mean_bonafide_score": counter["score_sum"] / total if total else math.nan,
        "min_bonafide_score": counter["score_min"] if total else math.nan,
        "max_bonafide_score": counter["score_max"] if total else math.nan,
    }


def score_rows(
    model_name: str,
    dataset: str,
    score_path: Path,
    labels: Dict[str, str],
    threshold: float,
) -> List[dict]:
    counters: Dict[str, Dict[str, float]] = defaultdict(init_counter)
    missing_labels = 0

    with score_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parsed = parse_score_line(raw_line)
            if parsed is None:
                continue
            filepath, _, bonafide_score, _ = parsed
            label = labels.get(filepath)
            if label is None:
                missing_labels += 1
                label = "spoof"
            model_type = filepath.split("/", 1)[0] if "/" in filepath else "(root)"
            update_counter(counters[model_type], label, float(bonafide_score), threshold)

    rows = [counter_to_row(model_name, dataset, model_type, counter) for model_type, counter in counters.items()]
    for row in rows:
        row["missing_protocol_labels"] = missing_labels
    return rows


def find_model_dirs(results_root: Path) -> Iterable[Path]:
    for child in sorted(results_root.iterdir()):
        if child.is_dir() and (child / "summary_results_details.jsonl").exists():
            yield child


def format_pct(value: float) -> str:
    return "NaN" if pd.isna(value) else f"{value:.2f}%"


def write_dataset_plot(dataset_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = dataset_df.sort_values(["accuracy_pct", "file_count"], ascending=[True, False])
    height = max(4.0, min(12.0, 0.34 * len(plot_df) + 1.8))
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(plot_df["model_type"], plot_df["accuracy_pct"], color="#2b6cb0")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Accuracy (%)")
    ax.set_ylabel("Model type")
    ax.set_title(dataset_df["dataset"].iloc[0])
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    view = df.loc[:, columns].copy()
    for col in ["accuracy_pct", "error_rate_pct", "mean_bonafide_score", "min_bonafide_score", "max_bonafide_score"]:
        if col in view.columns:
            if col.endswith("_pct"):
                view[col] = view[col].map(format_pct)
            else:
                view[col] = view[col].map(lambda value: "NaN" if pd.isna(value) else f"{value:.4f}")
    return view.to_markdown(index=False)


def write_report(df: pd.DataFrame, output_dir: Path, threshold_name: str, threshold_rows: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "model_type_metrics.csv", index=False)
    threshold_rows.to_csv(output_dir / "thresholds.csv", index=False)

    lines = [
        "# May 2026 Model-Type Accuracy Report",
        "",
        f"- Results root: `{df['results_root'].iloc[0]}`",
        f"- Threshold metric: `{threshold_name}`",
        "- Decision rule: `bonafide if bonafide_score >= threshold`; these two datasets are spoof-only.",
        "- Model type: first path segment in the score file.",
        "",
        "## Thresholds",
        "",
        markdown_table(threshold_rows, ["benchmark_model", "threshold", "threshold_source"]),
        "",
    ]

    summary = (
        df.groupby(["dataset"], as_index=False)
        .agg(file_count=("file_count", "sum"), correct=("correct", "sum"), errors=("errors", "sum"))
        .assign(accuracy_pct=lambda item: 100.0 * item["correct"] / item["file_count"])
        .sort_values("dataset")
    )
    lines.extend(["## Dataset Summary", "", markdown_table(summary, ["dataset", "file_count", "correct", "errors", "accuracy_pct"]), ""])

    for dataset in sorted(df["dataset"].unique()):
        dataset_df = df[df["dataset"] == dataset].sort_values(["accuracy_pct", "file_count"], ascending=[True, False])
        csv_path = output_dir / f"{slugify(dataset)}_model_type_metrics.csv"
        png_path = output_dir / f"{slugify(dataset)}_accuracy_by_model_type.png"
        dataset_df.to_csv(csv_path, index=False)
        write_dataset_plot(dataset_df, png_path)
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"![{dataset} accuracy by model type]({png_path.name})",
                "",
                markdown_table(
                    dataset_df,
                    [
                        "model_type",
                        "file_count",
                        "correct",
                        "errors",
                        "false_accepts",
                        "accuracy_pct",
                        "mean_bonafide_score",
                        "min_bonafide_score",
                        "max_bonafide_score",
                    ],
                ),
                "",
                f"CSV: `{csv_path.name}`",
                "",
            ]
        )

    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    datasets = args.dataset or DEFAULT_DATASETS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (results_root / f"model_type_report_{timestamp}")

    rows: List[dict] = []
    threshold_records: List[dict] = []
    for model_dir in find_model_dirs(results_root):
        details = load_summary_details(model_dir / "summary_results_details.jsonl")
        if details is None:
            continue
        threshold, source = threshold_info(details, args.threshold)
        threshold_records.append(
            {
                "benchmark_model": model_dir.name,
                "threshold": threshold,
                "threshold_source": source,
            }
        )
        stats_by_dataset = dataset_stats(details)
        for dataset in datasets:
            stats = stats_by_dataset.get(dataset)
            if not stats:
                print(f"skip: {model_dir.name} missing dataset {dataset}", file=sys.stderr)
                continue
            score_path = resolve_path(str(stats["score_file"]), Path.cwd())
            protocol_path = resolve_path(str(stats.get("protocol_file") or ""), Path.cwd())
            labels = load_protocol_labels(protocol_path)
            for row in score_rows(model_dir.name, dataset, score_path, labels, threshold):
                row["threshold"] = threshold
                row["threshold_source"] = source
                row["score_file"] = str(score_path)
                row["protocol_file"] = str(protocol_path)
                row["results_root"] = str(results_root)
                rows.append(row)

    if not rows:
        raise RuntimeError(f"No score rows found under {results_root} for datasets: {', '.join(datasets)}")

    df = pd.DataFrame(rows).sort_values(["dataset", "accuracy_pct", "file_count"], ascending=[True, True, False])
    threshold_df = pd.DataFrame(threshold_records)
    write_report(df, output_dir.resolve(), args.threshold, threshold_df)

    print(f"Wrote report: {output_dir.resolve() / 'README.md'}")
    print(f"Wrote metrics: {output_dir.resolve() / 'model_type_metrics.csv'}")
    print(f"Rows: {len(df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
