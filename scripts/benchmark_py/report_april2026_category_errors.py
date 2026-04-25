#!/usr/bin/env python3
"""Build a category-level error report for April 2026 benchmark results.

The report scans each model folder under a benchmark results root, infers a
meaningful dataset-specific category from file paths, applies the model's
global EER-threshold decision rule, and highlights the hardest categories by
accuracy and by raw misclassification count.

Notes:
- If a model folder contains `summary_results_details.jsonl`, the pooled
  `eer_threshold` from that file is used.
- Older runs without JSON details fall back to reconstructing a pooled
  EER-threshold from the merged protocol + merged score files.
- Category extraction rules are inferred from a small sample of paths per
  dataset, then applied consistently to all scored files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from benchmark_py.binary_eval import compute_eer
from benchmark_py.scores import parse_score_line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report hard subsets/categories for April 2026 benchmark runs")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("logs/results/April_2026_benchmark"),
        help="Folder containing one subfolder per benchmarked model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder for markdown/CSV report (default: <results-root>/category_subset_report)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Rows to keep in each leaderboard table",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum samples per category for leaderboard ranking",
    )
    return parser.parse_args()


def parse_protocol_line(line: str) -> Optional[Tuple[str, str, str]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = stripped.rsplit(maxsplit=2)
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def clean_part(text: str) -> str:
    return text.strip().strip('"').strip("'")


def split_filepath(filepath: str) -> List[str]:
    return [clean_part(part) for part in filepath.split("/") if part != ""]


def slugify_config(config_text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", config_text.strip()).strip("_")


def slugify_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()


def parse_summary_metadata(summary_path: Path) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    if not summary_path.exists():
        return metadata
    for raw_line in summary_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def pick_merged_file(model_dir: Path, prefix: str, config_slug: str) -> Path:
    candidates = sorted(model_dir.glob(f"{prefix}_*.txt"))
    if not candidates:
        raise FileNotFoundError(f"No {prefix}_*.txt found in {model_dir}")

    preferred = [path for path in candidates if config_slug and config_slug in path.name]
    pool = preferred or candidates
    # Prefer the shortest matching name to avoid accidental pooled/alternate variants.
    return sorted(pool, key=lambda item: (len(item.name), item.name))[0]


def load_threshold_from_details(details_path: Path) -> Optional[Dict[str, object]]:
    if not details_path.exists():
        return None

    for line in details_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if payload.get("type") != "pooled":
            continue
        results = payload.get("payload", {}).get("results", {})
        thresholds = results.get("validation", {}).get("thresholds", {})
        eer_threshold = thresholds.get("eer_threshold", {})
        value = eer_threshold.get("threshold")
        if value is None:
            continue
        return {
            "threshold": float(value),
            "source": str(eer_threshold.get("source") or "summary_results_details.jsonl"),
            "details_payload": payload,
        }

    return None


def load_protocol_labels(protocol_path: Path) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with protocol_path.open("r") as handle:
        for raw_line in handle:
            parsed = parse_protocol_line(raw_line)
            if parsed is None:
                continue
            filepath, _, label = parsed
            labels[filepath] = label
    return labels


def reconstruct_threshold(score_path: Path, labels: Dict[str, str]) -> float:
    y_true: List[int] = []
    scores: List[float] = []
    with score_path.open("r") as handle:
        for raw_line in handle:
            parsed = parse_score_line(raw_line)
            if parsed is None:
                continue
            filepath, _, bonafide_score, _ = parsed
            label = labels.get(filepath)
            if label is None:
                continue
            y_true.append(1 if label == "bonafide" else 0)
            scores.append(float(bonafide_score))

    eer, threshold = compute_eer(y_true, scores)
    if math.isnan(threshold):
        raise RuntimeError(f"Failed to reconstruct pooled EER threshold from {score_path}")
    return float(threshold)


def category_rule(dataset_name: str) -> Tuple[str, str, str]:
    if dataset_name == "2025_Kipot":
        return (
            "attack_code",
            "speaker_id",
            "Use segment 4 (`a00/a01/...`) as primary category and segment 5 as speaker.",
        )
    if dataset_name == "2026_April_Dataset_jiwon":
        return (
            "generator",
            "prompt_bucket",
            "Use segment 2 (`Seedance2/Sora/Veo3/WAN`) as primary category and segment 3 as prompt bucket.",
        )
    if dataset_name == "M-AILABS":
        return (
            "locale_speaker",
            "book_or_collection",
            "Use locale + speaker from segments 2 and 5 as primary category; segment 6 is the secondary book/collection hint.",
        )
    if dataset_name.startswith("MLAAD"):
        return (
            "tts_engine",
            "language",
            "Use segment 4 as TTS engine and segment 3 as language.",
        )
    return (
        "folder_1",
        "folder_2",
        "Fallback: use segment 2 as category and segment 3 as secondary facet.",
    )


def extract_category(filepath: str) -> Dict[str, str]:
    parts = split_filepath(filepath)
    dataset = parts[0] if parts else "unknown"
    category_type, facet_type, rule_text = category_rule(dataset)

    category = "unknown"
    facet = "unknown"

    if dataset == "2025_Kipot":
        if len(parts) > 3:
            category = parts[3]
        if len(parts) > 4:
            facet = parts[4]
    elif dataset == "2026_April_Dataset_jiwon":
        if len(parts) > 1:
            category = parts[1]
        if len(parts) > 2:
            facet = parts[2]
    elif dataset == "M-AILABS":
        locale = parts[1] if len(parts) > 1 else "unknown"
        speaker = parts[4] if len(parts) > 4 else (parts[3] if len(parts) > 3 else "unknown")
        category = f"{locale} / {speaker}"
        facet = parts[5] if len(parts) > 5 else (parts[2] if len(parts) > 2 else "unknown")
    elif dataset.startswith("MLAAD"):
        if len(parts) > 3:
            category = parts[3]
        if len(parts) > 2:
            facet = parts[2]
    else:
        if len(parts) > 1:
            category = parts[1]
        if len(parts) > 2:
            facet = parts[2]

    return {
        "dataset": dataset,
        "category_type": category_type,
        "category": category,
        "facet_type": facet_type,
        "facet": facet,
        "rule_text": rule_text,
    }


def init_counter() -> Dict[str, float]:
    return {
        "samples": 0,
        "n_bonafide": 0,
        "n_spoof": 0,
        "errors": 0,
        "false_accepts": 0,
        "false_rejects": 0,
        "score_sum": 0.0,
    }


def update_counter(counter: Dict[str, float], label: str, score: float, predicted_label: str) -> None:
    counter["samples"] += 1
    counter["score_sum"] += float(score)
    if label == "bonafide":
        counter["n_bonafide"] += 1
    else:
        counter["n_spoof"] += 1

    if predicted_label != label:
        counter["errors"] += 1
        if label == "spoof" and predicted_label == "bonafide":
            counter["false_accepts"] += 1
        if label == "bonafide" and predicted_label == "spoof":
            counter["false_rejects"] += 1


def finalize_records(
    model_name: str,
    threshold: float,
    threshold_source: str,
    counters: Dict[Tuple[str, ...], Dict[str, float]],
    key_names: Sequence[str],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for key, values in counters.items():
        samples = int(values["samples"])
        errors = int(values["errors"])
        accuracy = 100.0 * (1.0 - (errors / samples)) if samples else math.nan
        row = {
            "model": model_name,
            "threshold": float(threshold),
            "threshold_source": threshold_source,
            "samples": samples,
            "n_bonafide": int(values["n_bonafide"]),
            "n_spoof": int(values["n_spoof"]),
            "errors": errors,
            "false_accepts": int(values["false_accepts"]),
            "false_rejects": int(values["false_rejects"]),
            "accuracy_pct": accuracy,
            "error_rate_pct": 100.0 - accuracy if not math.isnan(accuracy) else math.nan,
            "mean_score": values["score_sum"] / samples if samples else math.nan,
        }
        row.update(dict(zip(key_names, key)))
        records.append(row)
    return records


@dataclass
class ModelRun:
    model_name: str
    config: str
    base_model_path: str
    summary_path: Path
    protocol_path: Path
    score_path: Path
    threshold: float
    threshold_source: str
    threshold_note: str
    overall_samples: int
    overall_errors: int
    overall_accuracy_pct: float


@dataclass
class SkippedRun:
    model_name: str
    reason: str


def load_model_run(model_dir: Path) -> Tuple[ModelRun, List[Dict[str, object]], List[Dict[str, object]], Dict[str, Dict[str, object]]]:
    summary_path = model_dir / "summary_results.txt"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary_results.txt in {model_dir}")

    summary_meta = parse_summary_metadata(summary_path)
    config_text = summary_meta.get("Config", "")
    config_slug = slugify_config(config_text)
    protocol_path = pick_merged_file(model_dir, "merged_protocol", config_slug)
    score_path = pick_merged_file(model_dir, "merged_scores", config_slug)

    labels = load_protocol_labels(protocol_path)
    if not labels:
        raise RuntimeError(f"No protocol labels parsed from {protocol_path.name}")

    threshold_info = load_threshold_from_details(model_dir / "summary_results_details.jsonl")
    if threshold_info is not None:
        threshold = float(threshold_info["threshold"])
        threshold_source = str(threshold_info["source"])
        threshold_note = "from summary_results_details.jsonl"
    else:
        threshold = reconstruct_threshold(score_path, labels)
        threshold_source = "reconstructed_test_eer"
        threshold_note = "JSON details missing; pooled EER threshold reconstructed from merged scores"

    primary_counters: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(init_counter)
    secondary_counters: Dict[Tuple[str, ...], Dict[str, float]] = defaultdict(init_counter)
    overall_counter = init_counter()
    dataset_samples: Dict[str, Dict[str, object]] = {}

    with score_path.open("r") as handle:
        for raw_line in handle:
            parsed = parse_score_line(raw_line)
            if parsed is None:
                continue

            filepath, _, bonafide_score, _ = parsed
            label = labels.get(filepath)
            if label is None:
                continue

            info = extract_category(filepath)
            dataset = info["dataset"]
            predicted_label = "bonafide" if float(bonafide_score) >= threshold else "spoof"

            update_counter(overall_counter, label, float(bonafide_score), predicted_label)

            primary_key = (dataset, info["category_type"], info["category"])
            secondary_key = (dataset, info["category_type"], info["category"], info["facet_type"], info["facet"])
            update_counter(primary_counters[primary_key], label, float(bonafide_score), predicted_label)
            update_counter(secondary_counters[secondary_key], label, float(bonafide_score), predicted_label)

            entry = dataset_samples.setdefault(
                dataset,
                {
                    "category_type": info["category_type"],
                    "facet_type": info["facet_type"],
                    "rule_text": info["rule_text"],
                    "sample_paths": [],
                },
            )
            sample_paths: List[str] = entry["sample_paths"]
            if len(sample_paths) < 3:
                sample_paths.append(filepath)

    if int(overall_counter["samples"]) == 0:
        raise RuntimeError(f"No scored rows matched protocol labels in {score_path.name}")

    primary_records = finalize_records(
        model_name=model_dir.name,
        threshold=threshold,
        threshold_source=threshold_source,
        counters=primary_counters,
        key_names=("dataset", "category_type", "category"),
    )
    secondary_records = finalize_records(
        model_name=model_dir.name,
        threshold=threshold,
        threshold_source=threshold_source,
        counters=secondary_counters,
        key_names=("dataset", "category_type", "category", "facet_type", "facet"),
    )

    overall_samples = int(overall_counter["samples"])
    overall_errors = int(overall_counter["errors"])
    overall_accuracy = 100.0 * (1.0 - (overall_errors / overall_samples)) if overall_samples else math.nan

    run = ModelRun(
        model_name=model_dir.name,
        config=config_text,
        base_model_path=summary_meta.get("Base_model_path", ""),
        summary_path=summary_path,
        protocol_path=protocol_path,
        score_path=score_path,
        threshold=threshold,
        threshold_source=threshold_source,
        threshold_note=threshold_note,
        overall_samples=overall_samples,
        overall_errors=overall_errors,
        overall_accuracy_pct=overall_accuracy,
    )
    return run, primary_records, secondary_records, dataset_samples


def summarize_primary(primary_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        primary_df.groupby(["dataset", "category_type", "category"], dropna=False)
        .agg(
            models_seen=("model", "nunique"),
            total_samples=("samples", "sum"),
            total_errors=("errors", "sum"),
            mean_accuracy_pct=("accuracy_pct", "mean"),
            min_accuracy_pct=("accuracy_pct", "min"),
            max_accuracy_pct=("accuracy_pct", "max"),
        )
        .reset_index()
    )
    if grouped.empty:
        return grouped

    worst_index = (
        primary_df.groupby(["dataset", "category_type", "category"], dropna=False)["accuracy_pct"]
        .idxmin()
        .dropna()
        .astype(int)
    )
    worst_rows = primary_df.loc[worst_index, ["dataset", "category_type", "category", "model", "accuracy_pct", "errors"]]
    worst_rows = worst_rows.rename(
        columns={
            "model": "worst_model",
            "accuracy_pct": "worst_model_accuracy_pct",
            "errors": "worst_model_errors",
        }
    )
    return grouped.merge(worst_rows, on=["dataset", "category_type", "category"], how="left")


def summarize_secondary(secondary_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        secondary_df.groupby(["dataset", "category_type", "category", "facet_type", "facet"], dropna=False)
        .agg(
            models_seen=("model", "nunique"),
            total_samples=("samples", "sum"),
            total_errors=("errors", "sum"),
            mean_accuracy_pct=("accuracy_pct", "mean"),
            min_accuracy_pct=("accuracy_pct", "min"),
        )
        .reset_index()
    )
    return grouped


def render_dataframe(df: pd.DataFrame, columns: Sequence[str], top_k: Optional[int] = None) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, list(columns)]
    if top_k is not None:
        view = view.head(top_k)
    return "```\n" + view.to_string(index=False) + "\n```"


def format_pct(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    return f"{float(value):.2f}%"


def save_horizontal_bar_chart(
    df: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    output_path: Path,
    color: str,
    annotate_pct: bool = False,
) -> None:
    if df.empty:
        return

    plot_df = df.copy()
    labels = plot_df[label_col].astype(str).tolist()
    values = plot_df[value_col].astype(float).tolist()

    fig_height = max(4.0, 0.55 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(labels, values, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    value_max = max(values) if values else 0.0
    pad = max(value_max * 0.01, 0.5)
    for bar, value in zip(bars, values):
        text = f"{value:.2f}%" if annotate_pct else f"{int(round(value))}"
        ax.text(value + pad, bar.get_y() + bar.get_height() / 2.0, text, va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_heatmap(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    top_k: int,
) -> None:
    if df.empty:
        return

    heatmap_df = df.copy()
    heatmap_df["label"] = heatmap_df["category"].astype(str)
    category_order = (
        heatmap_df.groupby("label", dropna=False)["accuracy_pct"]
        .mean()
        .sort_values()
        .head(top_k)
        .index
    )
    heatmap_df = heatmap_df[heatmap_df["label"].isin(category_order)]
    if heatmap_df.empty:
        return

    pivot = (
        heatmap_df.pivot_table(index="label", columns="model", values="accuracy_pct", aggfunc="mean")
        .reindex(category_order)
    )

    fig_height = max(4.0, 0.55 * len(pivot.index) + 1.5)
    fig_width = max(6.0, 1.5 * len(pivot.columns) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)

    ax.set_title(title)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(list(pivot.columns), rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))

    for row_idx in range(len(pivot.index)):
        for col_idx in range(len(pivot.columns)):
            value = pivot.iat[row_idx, col_idx]
            if pd.isna(value):
                text = "-"
            else:
                text = f"{value:.1f}"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color="black", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy (%)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_image_markdown(image_path: Path, alt_text: str) -> str:
    return f"![{alt_text}]({image_path.name})"


def build_dataset_index_rows(primary_df: pd.DataFrame, top_k: int, min_samples: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in sorted(primary_df["dataset"].dropna().unique()):
        dataset_df = primary_df[(primary_df["dataset"] == dataset) & (primary_df["samples"] >= min_samples)].copy()
        if dataset_df.empty:
            continue
        summary = summarize_primary(dataset_df).sort_values(
            by=["mean_accuracy_pct", "total_errors", "total_samples"],
            ascending=[True, False, False],
        )
        top_row = summary.iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "models_seen": int(dataset_df["model"].nunique()),
                "categories_seen": int(dataset_df["category"].nunique()),
                "hardest_category": top_row["category"],
                "hardest_mean_accuracy_pct": top_row["mean_accuracy_pct"],
                "hardest_total_errors": int(top_row["total_errors"]),
            }
        )
    return pd.DataFrame(rows).sort_values(by=["hardest_mean_accuracy_pct", "hardest_total_errors"], ascending=[True, False])


def build_dataset_report(
    dataset: str,
    dataset_dir: Path,
    dataset_primary_df: pd.DataFrame,
    dataset_secondary_df: pd.DataFrame,
    dataset_rule_row: pd.Series,
    top_k: int,
    min_samples: int,
) -> None:
    primary_summary = summarize_primary(dataset_primary_df)
    primary_summary = primary_summary[primary_summary["total_samples"] >= min_samples].copy()
    primary_summary = primary_summary.sort_values(
        by=["mean_accuracy_pct", "total_errors", "total_samples"],
        ascending=[True, False, False],
    )

    error_summary = primary_summary.sort_values(
        by=["total_errors", "mean_accuracy_pct", "total_samples"],
        ascending=[False, True, False],
    )

    secondary_summary = summarize_secondary(dataset_secondary_df)
    secondary_summary = secondary_summary[secondary_summary["total_samples"] >= min_samples].copy()
    secondary_summary = secondary_summary.sort_values(
        by=["mean_accuracy_pct", "total_errors", "total_samples"],
        ascending=[True, False, False],
    )

    accuracy_png = dataset_dir / "lowest_accuracy_categories.png"
    errors_png = dataset_dir / "highest_error_categories.png"
    heatmap_png = dataset_dir / "model_accuracy_heatmap.png"

    save_horizontal_bar_chart(
        df=primary_summary.head(top_k),
        label_col="category",
        value_col="mean_accuracy_pct",
        title=f"{dataset}: Lowest Mean Accuracy Categories",
        xlabel="Mean accuracy (%)",
        output_path=accuracy_png,
        color="#c44e52",
        annotate_pct=True,
    )
    save_horizontal_bar_chart(
        df=error_summary.head(top_k),
        label_col="category",
        value_col="total_errors",
        title=f"{dataset}: Highest Misclassification Counts",
        xlabel="Misclassifications",
        output_path=errors_png,
        color="#4c72b0",
        annotate_pct=False,
    )
    save_accuracy_heatmap(
        df=dataset_primary_df[dataset_primary_df["samples"] >= min_samples].copy(),
        output_path=heatmap_png,
        title=f"{dataset}: Accuracy by Model and Category",
        top_k=top_k,
    )

    per_model_worst = dataset_primary_df[dataset_primary_df["samples"] >= min_samples].copy()
    per_model_worst = per_model_worst.sort_values(by=["model", "accuracy_pct", "errors"], ascending=[True, True, False])

    report_lines = [
        f"# {dataset}",
        "",
        f"- Category type: `{dataset_rule_row['category_type']}`",
        f"- Secondary facet: `{dataset_rule_row['facet_type']}`",
        f"- Rule: {dataset_rule_row['rule_text']}",
        f"- Sample paths used to infer rule:",
        "```",
        str(dataset_rule_row["sample_paths"]),
        "```",
        "",
        "## Visualizations",
        "",
        render_image_markdown(accuracy_png, f"{dataset} lowest mean accuracy categories"),
        "",
        render_image_markdown(errors_png, f"{dataset} highest error categories"),
        "",
        render_image_markdown(heatmap_png, f"{dataset} model accuracy heatmap"),
        "",
        "## Hardest Categories",
        "",
        render_dataframe(
            primary_summary.assign(
                mean_accuracy_pct=primary_summary["mean_accuracy_pct"].map(format_pct),
                min_accuracy_pct=primary_summary["min_accuracy_pct"].map(format_pct),
                max_accuracy_pct=primary_summary["max_accuracy_pct"].map(format_pct),
                worst_model_accuracy_pct=primary_summary["worst_model_accuracy_pct"].map(format_pct),
            ),
            columns=[
                "category",
                "models_seen",
                "total_samples",
                "total_errors",
                "mean_accuracy_pct",
                "worst_model",
                "worst_model_accuracy_pct",
            ],
            top_k=top_k,
        ),
        "",
        "## Fine-Grained Slices",
        "",
        render_dataframe(
            secondary_summary.assign(
                mean_accuracy_pct=secondary_summary["mean_accuracy_pct"].map(format_pct),
                min_accuracy_pct=secondary_summary["min_accuracy_pct"].map(format_pct),
            ),
            columns=[
                "category",
                "facet_type",
                "facet",
                "models_seen",
                "total_samples",
                "total_errors",
                "mean_accuracy_pct",
            ],
            top_k=top_k,
        ),
        "",
        "## Per-Model Worst Categories",
        "",
        render_dataframe(
            per_model_worst.assign(
                accuracy_pct=per_model_worst["accuracy_pct"].map(format_pct),
                error_rate_pct=per_model_worst["error_rate_pct"].map(format_pct),
            ),
            columns=[
                "model",
                "category",
                "samples",
                "errors",
                "accuracy_pct",
                "false_accepts",
                "false_rejects",
            ],
            top_k=top_k * max(1, int(per_model_worst["model"].nunique())),
        ),
        "",
    ]

    (dataset_dir / "report.md").write_text("\n".join(report_lines) + "\n")
    dataset_primary_df.sort_values(by=["model", "accuracy_pct", "errors"], ascending=[True, True, False]).to_csv(
        dataset_dir / "category_metrics.csv",
        index=False,
    )
    dataset_secondary_df.sort_values(by=["model", "accuracy_pct", "errors"], ascending=[True, True, False]).to_csv(
        dataset_dir / "category_slice_metrics.csv",
        index=False,
    )


def build_report(
    runs: Sequence[ModelRun],
    skipped_runs: Sequence[SkippedRun],
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    dataset_rule_rows: pd.DataFrame,
    output_dir: Path,
    top_k: int,
    min_samples: int,
) -> str:
    run_rows = pd.DataFrame(
        [
            {
                "model": run.model_name,
                "overall_accuracy_pct": run.overall_accuracy_pct,
                "overall_errors": run.overall_errors,
                "overall_samples": run.overall_samples,
                "threshold": run.threshold,
                "threshold_source": run.threshold_source,
                "threshold_note": run.threshold_note,
            }
            for run in runs
        ]
    ).sort_values(by=["overall_accuracy_pct", "overall_errors"], ascending=[True, False])

    dataset_index = build_dataset_index_rows(primary_df=primary_df, top_k=top_k, min_samples=min_samples)
    dataset_links_df = dataset_index.copy()
    if not dataset_links_df.empty:
        dataset_links_df["report"] = dataset_links_df["dataset"].map(
            lambda name: f"datasets/{slugify_name(name)}/report.md"
        )
        dataset_links_df["hardest_mean_accuracy_pct"] = dataset_links_df["hardest_mean_accuracy_pct"].map(format_pct)

    sections = [
        "# April 2026 Benchmark Category Report Index",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Results root: `{runs[0].summary_path.parents[1] if runs else output_dir.parent}`",
        f"- Models analyzed: `{len(runs)}`",
        f"- Models skipped: `{len(skipped_runs)}`",
        f"- Ranking filter: categories with at least `{min_samples}` scored samples",
        f"- Decision rule: `bonafide if bonafide_score >= pooled_eer_threshold`",
        "",
        "## Skipped Models",
        "",
        render_dataframe(
            pd.DataFrame([{"model": item.model_name, "reason": item.reason} for item in skipped_runs]),
            columns=["model", "reason"],
        ),
        "",
        "## Model Overview",
        "",
        render_dataframe(
            run_rows.assign(
                overall_accuracy_pct=run_rows["overall_accuracy_pct"].map(lambda x: f"{x:.2f}%"),
                threshold=run_rows["threshold"].map(lambda x: f"{x:.6f}"),
            ),
            columns=[
                "model",
                "overall_accuracy_pct",
                "overall_errors",
                "overall_samples",
                "threshold",
                "threshold_source",
                "threshold_note",
            ],
        ),
        "",
        "## Dataset Reports",
        "",
        render_dataframe(
            dataset_links_df,
            columns=[
                "dataset",
                "models_seen",
                "categories_seen",
                "hardest_category",
                "hardest_mean_accuracy_pct",
                "hardest_total_errors",
                "report",
            ],
        ),
        "",
        "Each dataset now has its own compact report with plots under `datasets/<dataset>/report.md`.",
        "",
    ]

    sections.extend(
        [
            "## Notes",
            "",
            "- Main index is intentionally short; details moved into one report per dataset.",
            "- Each dataset report includes plots for lowest-accuracy categories, highest-error categories, and a model-by-category accuracy heatmap.",
            "- Category rules still come from a very small sample of scored paths, not by reading full raw dataset protocols manually.",
        ]
    )

    return "\n".join(sections) + "\n"


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_dir = (args.output_dir or (results_root / "category_subset_report")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted(path for path in results_root.iterdir() if path.is_dir())
    if not model_dirs:
        raise FileNotFoundError(f"No model folders found under {results_root}")

    runs: List[ModelRun] = []
    skipped_runs: List[SkippedRun] = []
    primary_records: List[Dict[str, object]] = []
    secondary_records: List[Dict[str, object]] = []
    dataset_rule_map: Dict[str, Dict[str, object]] = {}

    for model_dir in model_dirs:
        summary_path = model_dir / "summary_results.txt"
        if not summary_path.exists():
            continue

        try:
            run, run_primary, run_secondary, dataset_samples = load_model_run(model_dir)
        except Exception as exc:
            skipped_runs.append(SkippedRun(model_name=model_dir.name, reason=str(exc)))
            continue
        runs.append(run)
        primary_records.extend(run_primary)
        secondary_records.extend(run_secondary)

        for dataset, payload in dataset_samples.items():
            existing = dataset_rule_map.get(dataset)
            if existing is None:
                dataset_rule_map[dataset] = payload

    if not runs:
        raise RuntimeError(f"No valid model runs found under {results_root}")

    primary_df = pd.DataFrame(primary_records)
    secondary_df = pd.DataFrame(secondary_records)
    dataset_rule_rows = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "category_type": payload["category_type"],
                "facet_type": payload["facet_type"],
                "rule_text": payload["rule_text"],
                "sample_paths": "\n".join(payload["sample_paths"]),
            }
            for dataset, payload in sorted(dataset_rule_map.items())
        ]
    )

    primary_csv = output_dir / "category_metrics_by_model.csv"
    secondary_csv = output_dir / "category_slice_metrics_by_model.csv"
    primary_summary_csv = output_dir / "hardest_categories_overall.csv"
    report_md = output_dir / "subset_error_report.md"
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    primary_df.sort_values(by=["model", "accuracy_pct", "errors"], ascending=[True, True, False]).to_csv(primary_csv, index=False)
    secondary_df.sort_values(by=["model", "accuracy_pct", "errors"], ascending=[True, True, False]).to_csv(secondary_csv, index=False)
    summarize_primary(primary_df).sort_values(
        by=["mean_accuracy_pct", "total_errors", "total_samples"], ascending=[True, False, False]
    ).to_csv(primary_summary_csv, index=False)

    for dataset in sorted(primary_df["dataset"].dropna().unique()):
        dataset_primary_df = primary_df[primary_df["dataset"] == dataset].copy()
        dataset_secondary_df = secondary_df[secondary_df["dataset"] == dataset].copy()
        dataset_rule_row = dataset_rule_rows[dataset_rule_rows["dataset"] == dataset].iloc[0]
        dataset_dir = datasets_dir / slugify_name(dataset)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        build_dataset_report(
            dataset=dataset,
            dataset_dir=dataset_dir,
            dataset_primary_df=dataset_primary_df,
            dataset_secondary_df=dataset_secondary_df,
            dataset_rule_row=dataset_rule_row,
            top_k=args.top_k,
            min_samples=args.min_samples,
        )

    report_text = build_report(
        runs=runs,
        skipped_runs=skipped_runs,
        primary_df=primary_df,
        secondary_df=secondary_df,
        dataset_rule_rows=dataset_rule_rows,
        output_dir=output_dir,
        top_k=args.top_k,
        min_samples=args.min_samples,
    )
    report_md.write_text(report_text)

    print(f"Wrote: {report_md}")
    print(f"Wrote: {primary_csv}")
    print(f"Wrote: {secondary_csv}")
    print(f"Wrote: {primary_summary_csv}")
    print(f"Models analyzed: {len(runs)}")
    print(f"Models skipped: {len(skipped_runs)}")


if __name__ == "__main__":
    main()
