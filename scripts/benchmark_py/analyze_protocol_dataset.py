#!/usr/bin/env python3
"""Analyze protocol-only spoof/bonafide datasets and write Markdown reports."""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hungdx_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VALID_LABELS = {"bonafide", "spoof"}


@dataclass
class ProtocolEntry:
    dataset: str
    rel_path: str
    subset: str
    label: str
    category_type: str
    category: str
    facet_type: str
    facet: str
    ext: str
    depth: int
    exists: Optional[bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a protocol dataset pool and generate Markdown/PNG report")
    parser.add_argument("dataset_root", type=Path, help="Dataset root containing child protocol.txt files")
    parser.add_argument(
        "--reports-root",
        type=Path,
        default=Path("reports"),
        help="Parent folder where <timestamp>_report is created",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Rows shown in report tables and charts")
    parser.add_argument("--max-missing-check", type=int, default=300000, help="Max rows per dataset to check file existence")
    return parser.parse_args()


def parse_protocol_line(line: str) -> Optional[Tuple[str, str, str]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = stripped.rsplit(maxsplit=2)
    if len(parts) != 3:
        return None
    rel_path, subset, label = parts
    label = label.lower()
    if label not in VALID_LABELS:
        return None
    return rel_path.strip().strip('"').strip("'"), subset, label


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower() or "unknown"


def split_path(path: str) -> List[str]:
    return [part for part in path.split("/") if part]


def discover_protocols(root: Path) -> List[Tuple[str, Path, Path]]:
    child_protocols = sorted(path for path in root.glob("*/protocol.txt") if path.is_file())
    if child_protocols:
        return [(path.parent.name, path, path.parent) for path in child_protocols]
    protocol = root / "protocol.txt"
    if protocol.is_file():
        return [(root.name, protocol, root)]
    raise FileNotFoundError(f"No protocol.txt found under {root}")


def infer_category(dataset: str, rel_path: str) -> Tuple[str, str, str, str]:
    parts = split_path(rel_path)
    lower_dataset = dataset.lower()

    if dataset.startswith("MLAAD") or "mlaad" in lower_dataset:
        category = parts[3] if len(parts) > 3 else (parts[0] if parts else "unknown")
        facet = parts[2] if len(parts) > 2 else "unknown"
        return "tts_engine", category, "language", facet

    if "speechfake" in lower_dataset:
        category = parts[0] if len(parts) > 0 else "unknown"
        facet = parts[1] if len(parts) > 1 else "unknown"
        return "vocoder", category, "language", facet

    if "audioset" in lower_dataset:
        category = parts[0] if len(parts) > 0 else "unknown"
        facet = parts[1] if len(parts) > 1 else "unknown"
        return "collection", category, "folder", facet

    if "misclassified" in lower_dataset or (parts and parts[0].startswith("2026_April_Dataset")):
        category = parts[1] if len(parts) > 1 else (parts[0] if parts else "unknown")
        facet = parts[2] if len(parts) > 2 else "unknown"
        return "generator", category, "prompt_bucket", facet

    category = parts[0] if len(parts) > 0 else "unknown"
    facet = parts[1] if len(parts) > 1 else "unknown"
    return "folder_1", category, "folder_2", facet


def iter_entries(protocols: Sequence[Tuple[str, Path, Path]], max_missing_check: int) -> Iterable[ProtocolEntry]:
    checked_counts: Counter[str] = Counter()
    for dataset, protocol_path, base_dir in protocols:
        with protocol_path.open("r", errors="replace") as handle:
            for raw_line in handle:
                parsed = parse_protocol_line(raw_line)
                if parsed is None:
                    continue
                rel_path, subset, label = parsed
                parts = split_path(rel_path)
                category_type, category, facet_type, facet = infer_category(dataset, rel_path)
                ext = Path(rel_path).suffix.lower() or "<none>"
                checked_counts[dataset] += 1
                exists: Optional[bool] = None
                if checked_counts[dataset] <= max_missing_check:
                    exists = (base_dir / rel_path).exists()
                yield ProtocolEntry(
                    dataset=dataset,
                    rel_path=rel_path,
                    subset=subset,
                    label=label,
                    category_type=category_type,
                    category=category,
                    facet_type=facet_type,
                    facet=facet,
                    ext=ext,
                    depth=len(parts),
                    exists=exists,
                )


def entries_to_frame(entries: Iterable[ProtocolEntry]) -> pd.DataFrame:
    return pd.DataFrame([entry.__dict__ for entry in entries])


def pct(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    return f"{value:.2f}%"


def render_table(df: pd.DataFrame, columns: Sequence[str], top_k: Optional[int] = None) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, list(columns)].copy()
    if top_k is not None:
        view = view.head(top_k)
    return "```\n" + view.to_string(index=False) + "\n```"


def save_bar(df: pd.DataFrame, label_col: str, value_col: str, title: str, output: Path, color: str) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    labels = plot_df[label_col].astype(str).tolist()
    values = plot_df[value_col].astype(float).tolist()
    fig_height = max(4.0, 0.45 * len(plot_df) + 1.6)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(labels, values, color=color)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    pad = max(values) * 0.01 if values else 1
    for bar, value in zip(bars, values):
        ax.text(value + pad, bar.get_y() + bar.get_height() / 2, f"{int(value):,}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_stacked_label_chart(df: pd.DataFrame, output: Path) -> None:
    if df.empty:
        return
    pivot = df.pivot_table(index="dataset", columns="label", values="samples", aggfunc="sum", fill_value=0)
    for label in VALID_LABELS:
        if label not in pivot.columns:
            pivot[label] = 0
    pivot = pivot[["bonafide", "spoof"]].sort_values(by=["bonafide", "spoof"], ascending=False)
    fig_height = max(4.0, 0.45 * len(pivot.index) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    left = [0] * len(pivot.index)
    colors = {"bonafide": "#4c72b0", "spoof": "#c44e52"}
    for label in ["bonafide", "spoof"]:
        values = pivot[label].tolist()
        ax.barh(pivot.index, values, left=left, label=label, color=colors[label])
        left = [old + new for old, new in zip(left, values)]
    ax.set_title("Class Balance by Dataset")
    ax.set_xlabel("Samples")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(df: pd.DataFrame, output: Path, top_k: int) -> None:
    if df.empty:
        return
    top_categories = (
        df.groupby("category", dropna=False)["samples"].sum().sort_values(ascending=False).head(top_k).index
    )
    view = df[df["category"].isin(top_categories)]
    pivot = view.pivot_table(index="dataset", columns="category", values="samples", aggfunc="sum", fill_value=0)
    if pivot.empty:
        return
    fig_width = max(8.0, 0.7 * len(pivot.columns) + 3)
    fig_height = max(4.0, 0.5 * len(pivot.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    values = pivot.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    ax.set_title("Top Category Size Heatmap")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for row_idx in range(len(pivot.index)):
        for col_idx in range(len(pivot.columns)):
            ax.text(col_idx, row_idx, f"{int(values[row_idx, col_idx]):,}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Samples")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_pie(df: pd.DataFrame, output: Path) -> None:
    if df.empty:
        return
    counts = df.groupby("ext", dropna=False)["samples"].sum().sort_values(ascending=False)
    top = counts.head(8)
    rest = counts.iloc[8:].sum()
    if rest:
        top.loc["other"] = rest
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(top.values, labels=top.index, autopct="%1.1f%%", startangle=90)
    ax.set_title("Audio Extension Mix")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_dataset_report(dataset: str, dataset_df: pd.DataFrame, output_dir: Path, top_k: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset_df)
    label_counts = dataset_df["label"].value_counts()
    subset_counts = dataset_df["subset"].value_counts().reset_index()
    subset_counts.columns = ["subset", "samples"]
    category_counts = (
        dataset_df.groupby(["category_type", "category", "label"], dropna=False)
        .size()
        .reset_index(name="samples")
        .sort_values("samples", ascending=False)
    )
    category_totals = (
        dataset_df.groupby(["category_type", "category"], dropna=False)
        .agg(samples=("rel_path", "size"), bonafide=("label", lambda s: int((s == "bonafide").sum())), spoof=("label", lambda s: int((s == "spoof").sum())))
        .reset_index()
        .sort_values("samples", ascending=False)
    )
    facet_totals = (
        dataset_df.groupby(["category", "facet_type", "facet"], dropna=False)
        .agg(samples=("rel_path", "size"), bonafide=("label", lambda s: int((s == "bonafide").sum())), spoof=("label", lambda s: int((s == "spoof").sum())))
        .reset_index()
        .sort_values("samples", ascending=False)
    )
    checked_exists = dataset_df["exists"].dropna()
    missing_rate = 100.0 * (1.0 - float(checked_exists.mean())) if not checked_exists.empty else math.nan

    category_png = output_dir / "largest_categories.png"
    facet_png = output_dir / "largest_facets.png"
    save_bar(category_totals.head(top_k), "category", "samples", f"{dataset}: Largest Categories", category_png, "#4c72b0")
    save_bar(facet_totals.head(top_k), "facet", "samples", f"{dataset}: Largest Facets", facet_png, "#55a868")

    samples = "\n".join(dataset_df["rel_path"].head(8).tolist())
    lines = [
        f"# {dataset}",
        "",
        f"- Samples: `{total:,}`",
        f"- Bonafide: `{int(label_counts.get('bonafide', 0)):,}`",
        f"- Spoof: `{int(label_counts.get('spoof', 0)):,}`",
        f"- Subsets: `{dataset_df['subset'].nunique():,}`",
        f"- Categories: `{dataset_df['category'].nunique():,}`",
        f"- Category rule: `{dataset_df['category_type'].iloc[0]}` / `{dataset_df['facet_type'].iloc[0]}`",
        f"- File-existence checked rows: `{len(checked_exists):,}`",
        f"- File-existence missing rate in checked rows: `{pct(missing_rate)}`",
        "",
        "## Visualizations",
        "",
        "![Largest categories](largest_categories.png)",
        "",
        "![Largest facets](largest_facets.png)",
        "",
        "## Subsets",
        "",
        render_table(subset_counts, ["subset", "samples"], top_k),
        "",
        "## Largest Categories",
        "",
        render_table(category_totals, ["category_type", "category", "samples", "bonafide", "spoof"], top_k),
        "",
        "## Largest Fine-Grained Facets",
        "",
        render_table(facet_totals, ["category", "facet_type", "facet", "samples", "bonafide", "spoof"], top_k),
        "",
        "## Sample Paths",
        "",
        "```",
        samples,
        "```",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))
    category_counts.to_csv(output_dir / "category_label_counts.csv", index=False)
    facet_totals.to_csv(output_dir / "facet_counts.csv", index=False)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.reports_root / f"{timestamp}_report").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    protocols = discover_protocols(dataset_root)
    df = entries_to_frame(iter_entries(protocols, args.max_missing_check))
    if df.empty:
        raise RuntimeError(f"No valid protocol rows found under {dataset_root}")

    dataset_summary = (
        df.groupby(["dataset", "label"], dropna=False)
        .size()
        .reset_index(name="samples")
        .sort_values(["dataset", "label"])
    )
    dataset_totals = (
        df.groupby("dataset", dropna=False)
        .agg(
            samples=("rel_path", "size"),
            bonafide=("label", lambda s: int((s == "bonafide").sum())),
            spoof=("label", lambda s: int((s == "spoof").sum())),
            subsets=("subset", "nunique"),
            categories=("category", "nunique"),
            path_depth_min=("depth", "min"),
            path_depth_max=("depth", "max"),
            file_exists_checked=("exists", lambda s: int(s.notna().sum())),
            missing_checked=("exists", lambda s: int((s.dropna() == False).sum())),
        )
        .reset_index()
        .sort_values("samples", ascending=False)
    )
    category_totals = (
        df.groupby(["dataset", "category_type", "category"], dropna=False)
        .agg(samples=("rel_path", "size"), bonafide=("label", lambda s: int((s == "bonafide").sum())), spoof=("label", lambda s: int((s == "spoof").sum())))
        .reset_index()
        .sort_values("samples", ascending=False)
    )
    ext_counts = df.groupby("ext", dropna=False).size().reset_index(name="samples").sort_values("samples", ascending=False)
    duplicates = df["rel_path"].duplicated().sum()

    save_stacked_label_chart(dataset_summary, output_dir / "class_balance_by_dataset.png")
    save_bar(dataset_totals.head(args.top_k), "dataset", "samples", "Largest Datasets", output_dir / "largest_datasets.png", "#4c72b0")
    save_heatmap(category_totals, output_dir / "category_size_heatmap.png", args.top_k)
    save_pie(ext_counts, output_dir / "audio_extension_mix.png")

    datasets_dir = output_dir / "datasets"
    for dataset in sorted(df["dataset"].unique()):
        write_dataset_report(
            dataset=dataset,
            dataset_df=df[df["dataset"] == dataset].copy(),
            output_dir=datasets_dir / slugify(dataset),
            top_k=args.top_k,
        )

    dataset_totals.to_csv(output_dir / "dataset_summary.csv", index=False)
    category_totals.to_csv(output_dir / "category_summary.csv", index=False)
    ext_counts.to_csv(output_dir / "extension_summary.csv", index=False)

    lines = [
        "# Dataset Pool Analysis",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Dataset root: `{dataset_root}`",
        f"- Protocol files: `{len(protocols):,}`",
        f"- Total protocol rows: `{len(df):,}`",
        f"- Datasets: `{df['dataset'].nunique():,}`",
        f"- Bonafide: `{int((df['label'] == 'bonafide').sum()):,}`",
        f"- Spoof: `{int((df['label'] == 'spoof').sum()):,}`",
        f"- Duplicate relative paths: `{int(duplicates):,}`",
        "",
        "## Visualizations",
        "",
        "![Class balance by dataset](class_balance_by_dataset.png)",
        "",
        "![Largest datasets](largest_datasets.png)",
        "",
        "![Category size heatmap](category_size_heatmap.png)",
        "",
        "![Audio extension mix](audio_extension_mix.png)",
        "",
        "## Dataset Summary",
        "",
        render_table(dataset_totals, ["dataset", "samples", "bonafide", "spoof", "subsets", "categories", "path_depth_min", "path_depth_max", "missing_checked"], args.top_k),
        "",
        "## Largest Categories Overall",
        "",
        render_table(category_totals, ["dataset", "category_type", "category", "samples", "bonafide", "spoof"], args.top_k),
        "",
        "## Extension Mix",
        "",
        render_table(ext_counts, ["ext", "samples"], args.top_k),
        "",
        "## Per-Dataset Reports",
        "",
    ]
    for dataset in sorted(df["dataset"].unique()):
        lines.append(f"- [{dataset}](datasets/{slugify(dataset)}/report.md)")
    lines.append("")

    (output_dir / "README.md").write_text("\n".join(lines))
    print(f"Wrote: {output_dir / 'README.md'}")
    print(f"Wrote: {output_dir / 'dataset_summary.csv'}")
    print(f"Wrote: {output_dir / 'category_summary.csv'}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
