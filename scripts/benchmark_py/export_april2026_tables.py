#!/usr/bin/env python3
"""Export April 2026 benchmark tables to CSV files.

Inputs:
- Benchmark results root (contains one subfolder per model with summary_results.txt)
- Benchmark datasets root (used for context/validation in reports)

Outputs:
- summary_by_model_dataset.csv
- artificialanalysis_detailed.csv + artificialanalysis_generator_aggregate.csv
- intern_collect_detailed.csv + intern_collect_generator_aggregate.csv
- itw_real_collections_detailed.csv (bonafide-only slices by top-level source)
- spoof_collections_Hung_detailed.csv (spoof-only by TTS engine folder)
- 2026_April_Synthesizer_Hung_detailed.csv + 2026_April_Synthesizer_Hung_engine_aggregate.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

import eval_metrics_DF as em


def parse_protocol_line(line: str) -> Optional[Tuple[str, str, str]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.rsplit(maxsplit=2)
    if len(parts) != 3:
        return None
    file_id, subset, label = parts
    return file_id, subset, label


def parse_score_line(line: str) -> Optional[Tuple[str, float, float]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 3:
        return None
    try:
        spoof_score = float(parts[-2])
        bonafide_score = float(parts[-1])
    except ValueError:
        return None
    file_id = " ".join(parts[:-2])
    return file_id, spoof_score, bonafide_score


def resolve_path(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text.strip())
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def parse_summary_file(summary_path: Path) -> Tuple[Dict[str, str], pd.DataFrame]:
    metadata: Dict[str, str] = {}
    rows = []
    for raw_line in summary_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 6 and parts[0] not in {"Dataset", "POOLED_EER", "AVERAGE_EER"}:
                rows.append(
                    {
                        "dataset": parts[0],
                        "eer": _to_float(parts[1]),
                        "min_score": _to_float(parts[2]),
                        "max_score": _to_float(parts[3]),
                        "threshold": _to_float(parts[4]),
                        "accuracy": _to_float(parts[5]),
                    }
                )
        elif ":" in line:
            k, v = line.split(":", 1)
            metadata[k.strip()] = v.strip()
    return metadata, pd.DataFrame(rows)


def _to_float(x: str) -> float:
    try:
        value = float(x)
    except ValueError:
        return math.nan
    return value


def read_merged_tables(protocol_path: Path, scores_path: Path) -> pd.DataFrame:
    protocol_rows = []
    for line in protocol_path.read_text().splitlines():
        parsed = parse_protocol_line(line)
        if parsed:
            fpath, subset, label = parsed
            protocol_rows.append({"filepath": fpath, "subset": subset, "label": label})

    score_rows = []
    for line in scores_path.read_text().splitlines():
        parsed = parse_score_line(line)
        if parsed:
            fpath, spoof_score, score = parsed
            score_rows.append({"filepath": fpath, "spoof_score": spoof_score, "score": score})

    protocol_df = pd.DataFrame(protocol_rows)
    scores_df = pd.DataFrame(score_rows)
    merged = scores_df.merge(protocol_df, on="filepath", how="inner")
    merged["pred"] = merged.apply(
        lambda x: "bonafide" if x["spoof_score"] < x["score"] else "spoof", axis=1
    )
    return merged


def compute_group_metrics(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    records = []
    for keys, group in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: keys[idx] for idx, col in enumerate(group_cols)}
        rec["samples"] = int(len(group))
        rec["accuracy"] = float((group["pred"] == group["label"]).mean() * 100.0)
        rec["min_score"] = float(group["score"].min())
        rec["max_score"] = float(group["score"].max())

        bona = group.loc[group["label"] == "bonafide", "score"].to_numpy()
        spoof = group.loc[group["label"] == "spoof", "score"].to_numpy()
        if len(bona) > 0 and len(spoof) > 0:
            eer, threshold = em.compute_eer(bona, spoof)
            rec["eer"] = float(eer * 100.0)
            rec["threshold"] = float(threshold)
        else:
            rec["eer"] = math.nan
            rec["threshold"] = math.nan

        records.append(rec)
    return pd.DataFrame(records)


def collect_dataset_metrics(
    merged_df: pd.DataFrame,
    model_name: str,
    dataset_prefix: str,
    detail_levels: Iterable[Tuple[str, int]],
    aggregate_cols: Iterable[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dataset_df = merged_df[
        merged_df["filepath"].str.startswith(f"{dataset_prefix}/")
    ].copy()
    if dataset_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    path_parts = dataset_df["filepath"].str.split("/")
    dataset_df["model"] = model_name
    for column_name, path_idx in detail_levels:
        dataset_df[column_name] = path_parts.str[path_idx]

    detail_group_cols = ["model"] + [name for name, _ in detail_levels]
    agg_group_cols = ["model"] + list(aggregate_cols)
    detailed = compute_group_metrics(dataset_df, group_cols=detail_group_cols)
    aggregated = compute_group_metrics(dataset_df, group_cols=agg_group_cols)
    return detailed, aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Export April benchmark CSV tables")
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="Folder containing model result subfolders",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        required=True,
        help="Benchmark dataset root (for report context)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output CSV folder (default: <results-root>/csv_reports)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir or (args.results_root / "csv_reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    aa_detailed_frames = []
    aa_generator_frames = []
    intern_detailed_frames = []
    intern_generator_frames = []
    itw_detailed_frames = []
    spoof_hung_detailed_frames = []
    synth_hung_detailed_frames = []
    synth_hung_engine_frames = []

    model_dirs = sorted(p for p in args.results_root.iterdir() if p.is_dir())
    if not model_dirs:
        raise FileNotFoundError(f"No model folders found in {args.results_root}")

    for model_dir in model_dirs:
        summary_path = model_dir / "summary_results.txt"
        if not summary_path.exists():
            continue

        model_name = model_dir.name
        metadata, summary_df = parse_summary_file(summary_path)
        if not summary_df.empty:
            summary_df.insert(0, "model", model_name)
            summary_rows.append(summary_df)

        merged_protocol = metadata.get("MERGED_PROTOCOL")
        merged_scores = metadata.get("MERGED_SCORES")
        if not merged_protocol or not merged_scores:
            continue

        protocol_path = resolve_path(merged_protocol, repo_root)
        scores_path = resolve_path(merged_scores, repo_root)
        if not protocol_path.exists() or not scores_path.exists():
            continue

        merged_df = read_merged_tables(protocol_path, scores_path)
        aa_detailed, aa_generator = collect_dataset_metrics(
            merged_df=merged_df,
            model_name=model_name,
            dataset_prefix="artificialanalysis_audios",
            detail_levels=[("generator", 1), ("gender", 2), ("country", 3)],
            aggregate_cols=["generator"],
        )
        if not aa_detailed.empty:
            aa_detailed_frames.append(aa_detailed)
            aa_generator_frames.append(aa_generator)

        intern_detailed, intern_generator = collect_dataset_metrics(
            merged_df=merged_df,
            model_name=model_name,
            dataset_prefix="2026_April_Intern_collect",
            detail_levels=[("generator", 1), ("collection_id", 2)],
            aggregate_cols=["generator"],
        )
        if not intern_detailed.empty:
            intern_detailed_frames.append(intern_detailed)
            intern_generator_frames.append(intern_generator)

        itw_detailed, _ = collect_dataset_metrics(
            merged_df=merged_df,
            model_name=model_name,
            dataset_prefix="itw-real-collections",
            detail_levels=[("collection", 1)],
            aggregate_cols=["collection"],
        )
        if not itw_detailed.empty:
            itw_detailed_frames.append(itw_detailed)

        spoof_detailed, _ = collect_dataset_metrics(
            merged_df=merged_df,
            model_name=model_name,
            dataset_prefix="spoof-collections_Hung",
            detail_levels=[("engine", 1)],
            aggregate_cols=["engine"],
        )
        if not spoof_detailed.empty:
            spoof_hung_detailed_frames.append(spoof_detailed)

        synth_detailed, synth_engine = collect_dataset_metrics(
            merged_df=merged_df,
            model_name=model_name,
            dataset_prefix="2026_April_Synthesizer_Hung",
            detail_levels=[("engine", 1), ("mode", 2), ("variant", 3)],
            aggregate_cols=["engine"],
        )
        if not synth_detailed.empty:
            synth_hung_detailed_frames.append(synth_detailed)
            synth_hung_engine_frames.append(synth_engine)

    if summary_rows:
        summary_all = pd.concat(summary_rows, ignore_index=True)
    else:
        summary_all = pd.DataFrame(
            columns=["model", "dataset", "eer", "min_score", "max_score", "threshold", "accuracy"]
        )

    if aa_detailed_frames:
        detailed_all = pd.concat(aa_detailed_frames, ignore_index=True)
    else:
        detailed_all = pd.DataFrame(
            columns=[
                "model",
                "generator",
                "gender",
                "country",
                "samples",
                "accuracy",
                "eer",
                "threshold",
                "min_score",
                "max_score",
            ]
        )

    if aa_generator_frames:
        generator_all = pd.concat(aa_generator_frames, ignore_index=True)
    else:
        generator_all = pd.DataFrame(
            columns=[
                "model",
                "generator",
                "samples",
                "accuracy",
                "eer",
                "threshold",
                "min_score",
                "max_score",
            ]
        )

    if intern_detailed_frames:
        intern_detailed_all = pd.concat(intern_detailed_frames, ignore_index=True)
    else:
        intern_detailed_all = pd.DataFrame(
            columns=[
                "model",
                "generator",
                "collection_id",
                "samples",
                "accuracy",
                "eer",
                "threshold",
                "min_score",
                "max_score",
            ]
        )

    if intern_generator_frames:
        intern_generator_all = pd.concat(intern_generator_frames, ignore_index=True)
    else:
        intern_generator_all = pd.DataFrame(
            columns=[
                "model",
                "generator",
                "samples",
                "accuracy",
                "eer",
                "threshold",
                "min_score",
                "max_score",
            ]
        )

    if itw_detailed_frames:
        itw_detailed_all = pd.concat(itw_detailed_frames, ignore_index=True)
    else:
        itw_detailed_all = pd.DataFrame(
            columns=[
                "model",
                "collection",
                "samples",
                "accuracy",
                "eer",
                "threshold",
                "min_score",
                "max_score",
            ]
        )

    if spoof_hung_detailed_frames:
        spoof_hung_detailed_all = pd.concat(spoof_hung_detailed_frames, ignore_index=True)
    else:
        spoof_hung_detailed_all = pd.DataFrame(
            columns=[
                "model",
                "engine",
                "samples",
                "accuracy",
                "eer",
                "threshold",
                "min_score",
                "max_score",
            ]
        )

    if synth_hung_detailed_frames:
        synth_hung_detailed_all = pd.concat(synth_hung_detailed_frames, ignore_index=True)
    else:
        synth_hung_detailed_all = pd.DataFrame(
            columns=[
                "model",
                "engine",
                "mode",
                "variant",
                "samples",
                "accuracy",
                "eer",
                "threshold",
                "min_score",
                "max_score",
            ]
        )

    if synth_hung_engine_frames:
        synth_hung_engine_all = pd.concat(synth_hung_engine_frames, ignore_index=True)
    else:
        synth_hung_engine_all = pd.DataFrame(
            columns=[
                "model",
                "engine",
                "samples",
                "accuracy",
                "eer",
                "threshold",
                "min_score",
                "max_score",
            ]
        )

    summary_out = output_dir / "summary_by_model_dataset.csv"
    detailed_out = output_dir / "artificialanalysis_detailed.csv"
    generator_out = output_dir / "artificialanalysis_generator_aggregate.csv"
    intern_detailed_out = output_dir / "intern_collect_detailed.csv"
    intern_generator_out = output_dir / "intern_collect_generator_aggregate.csv"
    itw_out = output_dir / "itw_real_collections_detailed.csv"
    spoof_hung_out = output_dir / "spoof_collections_Hung_detailed.csv"
    synth_hung_detailed_out = output_dir / "2026_April_Synthesizer_Hung_detailed.csv"
    synth_hung_engine_out = output_dir / "2026_April_Synthesizer_Hung_engine_aggregate.csv"

    summary_all.to_csv(summary_out, index=False)
    detailed_all.to_csv(detailed_out, index=False)
    generator_all.to_csv(generator_out, index=False)
    intern_detailed_all.to_csv(intern_detailed_out, index=False)
    intern_generator_all.to_csv(intern_generator_out, index=False)
    itw_detailed_all.to_csv(itw_out, index=False)
    spoof_hung_detailed_all.to_csv(spoof_hung_out, index=False)
    synth_hung_detailed_all.to_csv(synth_hung_detailed_out, index=False)
    synth_hung_engine_all.to_csv(synth_hung_engine_out, index=False)

    print(f"Benchmark root: {args.benchmark_root}")
    print(f"Wrote: {summary_out}")
    print(f"Wrote: {detailed_out}")
    print(f"Wrote: {generator_out}")
    print(f"Wrote: {intern_detailed_out}")
    print(f"Wrote: {intern_generator_out}")
    print(f"Wrote: {itw_out}")
    print(f"Wrote: {spoof_hung_out}")
    print(f"Wrote: {synth_hung_detailed_out}")
    print(f"Wrote: {synth_hung_engine_out}")
    print(
        "Rows => "
        f"summary: {len(summary_all)}, "
        f"aa_detailed: {len(detailed_all)}, "
        f"aa_generator: {len(generator_all)}, "
        f"intern_detailed: {len(intern_detailed_all)}, "
        f"intern_generator: {len(intern_generator_all)}, "
        f"itw_detailed: {len(itw_detailed_all)}, "
        f"spoof_hung_detailed: {len(spoof_hung_detailed_all)}, "
        f"synth_hung_detailed: {len(synth_hung_detailed_all)}, "
        f"synth_hung_engine: {len(synth_hung_engine_all)}"
    )


if __name__ == "__main__":
    main()
