"""
Binary spoof/bonafide evaluation helpers.

Assumptions:
- bonafide label = 1
- spoof label = 0
- higher score means more bonafide
- bonafide decision if score >= threshold
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from benchmark_py.protocol import parse_protocol_line
from benchmark_py.scores import parse_score_line


LABEL_TO_INT = {
    "bonafide": 1,
    "bona_fide": 1,
    "bona-fide": 1,
    "bona fide": 1,
    "human": 1,
    "real": 1,
    "true": 1,
    "1": 1,
    "spoof": 0,
    "fake": 0,
    "false": 0,
    "0": 0,
}


def load_eval_config(eval_config_path: Optional[Path]) -> Dict[str, Any]:
    if eval_config_path is None:
        return {}
    config_path = Path(eval_config_path)
    if not config_path.exists():
        return {}
    loaded = OmegaConf.load(config_path)
    container = OmegaConf.to_container(loaded, resolve=True)
    return container if isinstance(container, dict) else {}


def _normalize_fill_policy(fill_policy: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[str]]]:
    policy = fill_policy or {}
    normalized: Dict[str, Dict[str, Optional[str]]] = {}
    for dataset_name, dataset_policy in policy.items():
        if not isinstance(dataset_policy, dict):
            continue
        normalized[str(dataset_name)] = {
            "bonafide_source": dataset_policy.get("bonafide_source"),
            "spoof_source": dataset_policy.get("spoof_source"),
        }
    return normalized


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return math.nan
    return float(numerator / denominator)


def _safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _format_float(value: Any, decimals: int = 6) -> str:
    value = _safe_float(value)
    if math.isnan(value):
        return "NaN"
    return f"{value:.{decimals}f}"


def _format_percent(value: Any) -> str:
    value = _safe_float(value)
    if math.isnan(value):
        return "NaN"
    return f"{value * 100.0:.2f}%"


def _nan_threshold_metrics(reason: str, threshold: float) -> Dict[str, Any]:
    return {
        "threshold": threshold,
        "accuracy": math.nan,
        "bonafide_accuracy": math.nan,
        "spoof_accuracy": math.nan,
        "precision": math.nan,
        "recall": math.nan,
        "f1": math.nan,
        "far": math.nan,
        "frr": math.nan,
        "mdr": math.nan,
        "reason": reason,
        "tp": math.nan,
        "tn": math.nan,
        "fp": math.nan,
        "fn": math.nan,
    }


def normalize_label(label: Any) -> int:
    key = str(label).strip().lower()
    if key not in LABEL_TO_INT:
        raise ValueError(f"Unsupported label: {label}")
    return LABEL_TO_INT[key]


def load_protocol_dataframe(
    protocol_file: Path,
    preferred_subset: Optional[str] = None,
    fallback_to_all: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    with open(protocol_file, "r") as handle:
        for line in handle:
            parsed = parse_protocol_line(line)
            if parsed is None:
                continue
            filename, subset, label_name = parsed
            rows.append(
                {
                    "filename": filename,
                    "subset": subset,
                    "label_name": label_name,
                    "label": normalize_label(label_name),
                }
            )

    frame = pd.DataFrame(rows, columns=["filename", "subset", "label_name", "label"])
    if frame.empty or not preferred_subset:
        return frame

    preferred_mask = frame["subset"] == preferred_subset
    if preferred_mask.any():
        return frame.loc[preferred_mask].reset_index(drop=True)
    if fallback_to_all:
        return frame
    return frame.iloc[0:0].copy()


def load_score_dataframe(score_file: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    with open(score_file, "r") as handle:
        for line in handle:
            parsed = parse_score_line(line)
            if parsed is None:
                continue
            filename, spoof_score, bonafide_score, _ = parsed
            rows.append(
                {
                    "filename": filename,
                    "spoof_score": float(spoof_score),
                    "score": float(bonafide_score),
                }
            )

    frame = pd.DataFrame(rows, columns=["filename", "spoof_score", "score"])
    if frame.empty:
        return frame
    return frame.drop_duplicates(subset="filename", keep="last").reset_index(drop=True)


def build_eval_frame(
    score_file: Path,
    protocol_file: Path,
    dataset_name: Optional[str] = None,
    preferred_subset: Optional[str] = None,
    fallback_to_all: bool = True,
) -> pd.DataFrame:
    protocol_df = load_protocol_dataframe(
        protocol_file=protocol_file,
        preferred_subset=preferred_subset,
        fallback_to_all=fallback_to_all,
    )
    score_df = load_score_dataframe(score_file)

    merged = protocol_df.merge(score_df, on="filename", how="inner")
    merged["dataset_name"] = dataset_name or protocol_file.parent.name or protocol_file.stem
    merged = merged.loc[np.isfinite(merged["score"])].reset_index(drop=True)
    return merged[
        ["filename", "subset", "dataset_name", "label_name", "label", "spoof_score", "score"]
    ].copy()


def _prepare_arrays(
    labels: Sequence[int],
    scores: Sequence[float],
    sample_weight: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    labels_array = np.asarray(labels, dtype=int)
    scores_array = np.asarray(scores, dtype=float)

    mask = np.isfinite(scores_array)
    weight_array = None
    if sample_weight is not None:
        weight_array = np.asarray(sample_weight, dtype=float)
        mask &= np.isfinite(weight_array)
        mask &= weight_array >= 0.0

    labels_array = labels_array[mask]
    scores_array = scores_array[mask]
    if weight_array is not None:
        weight_array = weight_array[mask]

    return labels_array, scores_array, weight_array


def _binary_metric_reason(labels: Sequence[int]) -> Optional[str]:
    labels_array = np.asarray(labels, dtype=int)
    if labels_array.size == 0:
        return "no scored samples matched the protocol"

    unique_labels = np.unique(labels_array)
    if unique_labels.size < 2:
        label_name = "bonafide" if unique_labels[0] == 1 else "spoof"
        return f"only one class is present ({label_name}); EER/AUC require both bonafide and spoof samples"

    return None


def _roc_operating_points(
    labels: Sequence[int],
    scores: Sequence[float],
    sample_weight: Optional[Sequence[float]] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    labels_array, scores_array, weight_array = _prepare_arrays(labels, scores, sample_weight)
    reason = _binary_metric_reason(labels_array)
    if reason is not None:
        return None, None, None, reason

    fpr, tpr, thresholds = roc_curve(
        labels_array,
        scores_array,
        pos_label=1,
        sample_weight=weight_array,
        drop_intermediate=False,
    )
    frr = 1.0 - tpr
    return fpr, frr, thresholds, None


def compute_eer(
    labels: Sequence[int],
    scores: Sequence[float],
    sample_weight: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    fpr, frr, thresholds, reason = _roc_operating_points(labels, scores, sample_weight)
    if reason is not None:
        return math.nan, math.nan

    abs_diff = np.abs(fpr - frr)
    best_indices = np.flatnonzero(abs_diff == abs_diff.min())
    finite_indices = best_indices[np.isfinite(thresholds[best_indices])]
    chosen_index = int(finite_indices[-1] if finite_indices.size else best_indices[-1])

    eer = float((fpr[chosen_index] + frr[chosen_index]) / 2.0)
    threshold = float(thresholds[chosen_index]) if np.isfinite(thresholds[chosen_index]) else math.nan
    return eer, threshold


def compute_auc(
    labels: Sequence[int],
    scores: Sequence[float],
    sample_weight: Optional[Sequence[float]] = None,
) -> float:
    labels_array, scores_array, weight_array = _prepare_arrays(labels, scores, sample_weight)
    reason = _binary_metric_reason(labels_array)
    if reason is not None:
        return math.nan

    try:
        return float(roc_auc_score(labels_array, scores_array, sample_weight=weight_array))
    except ValueError:
        return math.nan


def _select_threshold_row(
    labels: Sequence[int],
    scores: Sequence[float],
    sample_weight: Optional[Sequence[float]] = None,
    mode: str = "eer",
    target_far: float = 0.01,
) -> Dict[str, Any]:
    fpr, frr, thresholds, reason = _roc_operating_points(labels, scores, sample_weight)
    if reason is not None:
        return {
            "threshold": math.nan,
            "far": math.nan,
            "frr": math.nan,
            "eer": math.nan,
            "reason": reason,
        }

    finite_mask = np.isfinite(thresholds)
    if mode == "eer":
        abs_diff = np.abs(fpr - frr)
        best_indices = np.flatnonzero(abs_diff == abs_diff.min())
        finite_indices = best_indices[finite_mask[best_indices]]
        chosen_index = int(finite_indices[-1] if finite_indices.size else best_indices[-1])
    elif mode == "target_far":
        eligible = np.flatnonzero((fpr <= target_far) & finite_mask)
        if eligible.size:
            chosen_index = int(eligible[-1])
        else:
            finite_indices = np.flatnonzero(finite_mask)
            chosen_index = int(finite_indices[0] if finite_indices.size else 0)
    else:
        raise ValueError(f"Unsupported threshold selection mode: {mode}")

    threshold = float(thresholds[chosen_index]) if np.isfinite(thresholds[chosen_index]) else math.nan
    return {
        "threshold": threshold,
        "far": float(fpr[chosen_index]),
        "frr": float(frr[chosen_index]),
        "eer": float((fpr[chosen_index] + frr[chosen_index]) / 2.0),
        "reason": None,
    }


def find_threshold_at_target_far(
    labels: Sequence[int],
    scores: Sequence[float],
    target_far: float = 0.01,
    sample_weight: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    result = _select_threshold_row(
        labels=labels,
        scores=scores,
        sample_weight=sample_weight,
        mode="target_far",
        target_far=target_far,
    )
    result["target_far"] = float(target_far)
    return result


def _find_threshold_at_eer(
    labels: Sequence[int],
    scores: Sequence[float],
    sample_weight: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    result = _select_threshold_row(
        labels=labels,
        scores=scores,
        sample_weight=sample_weight,
        mode="eer",
    )
    return result


def _find_best_f1_threshold(
    labels: Sequence[int],
    scores: Sequence[float],
    sample_weight: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    labels_array, scores_array, weight_array = _prepare_arrays(labels, scores, sample_weight)
    reason = _binary_metric_reason(labels_array)
    if reason is not None:
        return {
            "threshold": math.nan,
            "far": math.nan,
            "frr": math.nan,
            "f1": math.nan,
            "reason": reason,
        }

    precision, recall, thresholds = precision_recall_curve(
        labels_array,
        scores_array,
        sample_weight=weight_array,
        pos_label=1,
    )
    if thresholds.size == 0:
        return {
            "threshold": math.nan,
            "far": math.nan,
            "frr": math.nan,
            "f1": math.nan,
            "reason": "not enough score variation to search for a best-F1 threshold",
        }

    precision = precision[:-1]
    recall = recall[:-1]
    f1_scores = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.full_like(precision, np.nan, dtype=float),
        where=(precision + recall) > 0,
    )

    valid = np.flatnonzero(np.isfinite(f1_scores) & np.isfinite(thresholds))
    if valid.size == 0:
        return {
            "threshold": math.nan,
            "far": math.nan,
            "frr": math.nan,
            "f1": math.nan,
            "reason": "could not determine a finite best-F1 threshold",
        }

    best_index = int(valid[np.nanargmax(f1_scores[valid])])
    threshold = float(thresholds[best_index])
    metrics = compute_threshold_metrics(
        labels=labels_array,
        scores=scores_array,
        threshold=threshold,
        sample_weight=weight_array,
    )
    return {
        "threshold": threshold,
        "far": metrics["far"],
        "frr": metrics["frr"],
        "f1": float(f1_scores[best_index]),
        "reason": None,
    }


def compute_threshold_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    threshold: float,
    sample_weight: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    if not np.isfinite(threshold):
        return _nan_threshold_metrics("threshold is not available", _safe_float(threshold))

    labels_array, scores_array, weight_array = _prepare_arrays(labels, scores, sample_weight)
    if labels_array.size == 0:
        return _nan_threshold_metrics("no scored samples matched the protocol", threshold)

    weights = weight_array if weight_array is not None else np.ones_like(scores_array, dtype=float)
    pred_positive = scores_array >= threshold
    positive = labels_array == 1
    negative = labels_array == 0

    tp = float(weights[pred_positive & positive].sum())
    tn = float(weights[(~pred_positive) & negative].sum())
    fp = float(weights[pred_positive & negative].sum())
    fn = float(weights[(~pred_positive) & positive].sum())

    accuracy = _safe_ratio(tp + tn, tp + tn + fp + fn)
    bonafide_accuracy = _safe_ratio(tp, tp + fn)
    spoof_accuracy = _safe_ratio(tn, tn + fp)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    f1 = _safe_ratio(2.0 * precision * recall, precision + recall) if np.isfinite(precision + recall) else math.nan
    far = _safe_ratio(fp, fp + tn)
    frr = _safe_ratio(fn, fn + tp)
    mdr = frr

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "bonafide_accuracy": bonafide_accuracy,
        "spoof_accuracy": spoof_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far": far,
        "frr": frr,
        "mdr": mdr,
        "reason": None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_score_comparison_metrics(
    frame: pd.DataFrame,
    sample_weight: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    if frame.empty:
        return {
            "accuracy": math.nan,
            "bonafide_accuracy": math.nan,
            "spoof_accuracy": math.nan,
        }

    pred = (frame["score"].to_numpy(dtype=float) >= frame["spoof_score"].to_numpy(dtype=float)).astype(int)
    label = frame["label"].to_numpy(dtype=int)
    weights = (
        np.asarray(sample_weight, dtype=float)
        if sample_weight is not None
        else np.ones_like(label, dtype=float)
    )
    positive = label == 1
    negative = label == 0
    correct = pred == label

    return {
        "accuracy": _safe_ratio(float(weights[correct].sum()), float(weights.sum())),
        "bonafide_accuracy": _safe_ratio(float(weights[correct & positive].sum()), float(weights[positive].sum())),
        "spoof_accuracy": _safe_ratio(float(weights[correct & negative].sum()), float(weights[negative].sum())),
    }


def _score_range(frame: pd.DataFrame) -> Tuple[float, float]:
    if frame.empty:
        return math.nan, math.nan
    return float(frame["score"].min()), float(frame["score"].max())


def _class_name(label_value: int) -> str:
    return "bonafide" if int(label_value) == 1 else "spoof"


def _complete_missing_classes(
    dataset_frame: pd.DataFrame,
    all_frames: pd.DataFrame,
    fill_policy: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dataset_name = dataset_frame["dataset_name"].iloc[0] if not dataset_frame.empty else "unknown"
    normalized_policy = _normalize_fill_policy(fill_policy)
    dataset_policy = normalized_policy.get(str(dataset_name), {})

    completed = dataset_frame.copy()
    completed["is_fill_sample"] = False
    completed["fill_source_dataset"] = None

    info: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "used_fill": False,
        "filled_counts": {"bonafide": 0, "spoof": 0},
        "fill_sources": {
            "bonafide": dataset_policy.get("bonafide_source"),
            "spoof": dataset_policy.get("spoof_source"),
        },
        "missing_before": [],
        "missing_after": [],
        "notes": [],
        "effective_n_bonafide": int((dataset_frame["label"] == 1).sum()) if not dataset_frame.empty else 0,
        "effective_n_spoof": int((dataset_frame["label"] == 0).sum()) if not dataset_frame.empty else 0,
    }

    existing_labels = set(dataset_frame["label"].unique().tolist()) if not dataset_frame.empty else set()
    for label_value, policy_key in ((1, "bonafide_source"), (0, "spoof_source")):
        class_name = _class_name(label_value)
        if label_value in existing_labels:
            continue

        info["missing_before"].append(class_name)
        source_dataset = dataset_policy.get(policy_key)
        if not source_dataset:
            info["notes"].append(f"missing {class_name}; no fill source configured")
            continue

        source_rows = all_frames[
            (all_frames["dataset_name"] == source_dataset) & (all_frames["label"] == label_value)
        ].copy()
        if source_rows.empty:
            info["notes"].append(f"missing {class_name}; source {source_dataset} has no {_class_name(label_value)}")
            continue

        source_rows["dataset_name"] = dataset_name
        source_rows["is_fill_sample"] = True
        source_rows["fill_source_dataset"] = source_dataset
        completed = pd.concat([completed, source_rows], ignore_index=True)
        info["used_fill"] = True
        info["filled_counts"][class_name] = int(len(source_rows))
        info["notes"].append(f"filled {class_name} from {source_dataset} (+{len(source_rows)})")

    effective_labels = set(completed["label"].unique().tolist()) if not completed.empty else set()
    for label_value in (1, 0):
        if label_value not in effective_labels:
            info["missing_after"].append(_class_name(label_value))

    info["effective_n_bonafide"] = int((completed["label"] == 1).sum()) if not completed.empty else 0
    info["effective_n_spoof"] = int((completed["label"] == 0).sum()) if not completed.empty else 0
    return completed.reset_index(drop=True), info


def _evaluate_scope(
    frame: pd.DataFrame,
    thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
    sample_weight: Optional[Sequence[float]] = None,
    scope_name: Optional[str] = None,
    threshold_free_frame: Optional[pd.DataFrame] = None,
    threshold_free_note: Optional[str] = None,
    completion_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    thresholds = thresholds or {}

    threshold_free_frame = threshold_free_frame if threshold_free_frame is not None else frame
    tf_labels = threshold_free_frame["label"].to_numpy(dtype=int) if not threshold_free_frame.empty else np.asarray([], dtype=int)
    tf_scores = threshold_free_frame["score"].to_numpy(dtype=float) if not threshold_free_frame.empty else np.asarray([], dtype=float)
    reason = _binary_metric_reason(tf_labels)
    eer, eer_threshold = compute_eer(tf_labels, tf_scores, sample_weight)
    auc = compute_auc(tf_labels, tf_scores, sample_weight)
    min_score, max_score = _score_range(frame)

    labels = frame["label"].to_numpy(dtype=int) if not frame.empty else np.asarray([], dtype=int)
    scores = frame["score"].to_numpy(dtype=float) if not frame.empty else np.asarray([], dtype=float)
    legacy_metrics = compute_legacy_compatibility_metrics(frame)
    score_comparison_metrics = compute_score_comparison_metrics(frame, sample_weight=sample_weight)
    threshold_metrics: Dict[str, Dict[str, Any]] = {}
    for threshold_name, threshold_info in thresholds.items():
        threshold_value = _safe_float(threshold_info.get("threshold"))
        metrics = compute_threshold_metrics(labels, scores, threshold_value, sample_weight=sample_weight)
        metrics["selection"] = {
            "source": threshold_info.get("source"),
            "selection_metric": threshold_name,
            "validation_far": threshold_info.get("validation_far"),
            "validation_frr": threshold_info.get("validation_frr"),
            "validation_eer": threshold_info.get("validation_eer"),
            "target_far": threshold_info.get("target_far"),
            "validation_f1": threshold_info.get("validation_f1"),
            "selection_reason": threshold_info.get("reason"),
        }
        threshold_metrics[threshold_name] = metrics

    return {
        "dataset_name": scope_name or (frame["dataset_name"].iloc[0] if not frame.empty else "unknown"),
        "n_samples": int(len(frame)),
        "n_bonafide": int((frame["label"] == 1).sum()) if not frame.empty else 0,
        "n_spoof": int((frame["label"] == 0).sum()) if not frame.empty else 0,
        "min_score": min_score,
        "max_score": max_score,
        "legacy_accuracy": legacy_metrics.get("accuracy"),
        "threshold_free": {
            "eer": eer,
            "eer_threshold": eer_threshold,
            "roc_auc": auc,
            "accuracy": score_comparison_metrics["accuracy"],
            "bonafide_accuracy": score_comparison_metrics["bonafide_accuracy"],
            "spoof_accuracy": score_comparison_metrics["spoof_accuracy"],
            "reason": reason,
            "note": threshold_free_note,
        },
        "threshold_metrics": threshold_metrics,
        "completion": completion_info or {},
    }


def evaluate_per_dataset(
    frame: pd.DataFrame,
    thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
    fill_policy: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
) -> List[Dict[str, Any]]:
    if frame.empty:
        return []

    results: List[Dict[str, Any]] = []
    for dataset_name, dataset_frame in frame.groupby("dataset_name", sort=True):
        dataset_frame = dataset_frame.reset_index(drop=True)
        completed_frame, completion_info = _complete_missing_classes(dataset_frame, frame, fill_policy=fill_policy)
        note = "; ".join(completion_info["notes"]) if completion_info.get("notes") else None
        results.append(
            _evaluate_scope(
                dataset_frame,
                thresholds,
                scope_name=dataset_name,
                threshold_free_frame=completed_frame,
                threshold_free_note=note,
                completion_info=completion_info,
            )
        )
    return results


def evaluate_pooled(
    frame: pd.DataFrame,
    thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return _evaluate_scope(frame.reset_index(drop=True), thresholds, scope_name="raw_pooled")


def evaluate_balanced_pooled(
    frame: pd.DataFrame,
    thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if frame.empty:
        result = _evaluate_scope(frame.reset_index(drop=True), thresholds, scope_name="balanced_pooled")
        result["weighting"] = "equal_dataset_contribution"
        return result

    dataset_counts = frame["dataset_name"].value_counts()
    sample_weight = frame["dataset_name"].map(lambda name: 1.0 / float(dataset_counts[name])).to_numpy(dtype=float)
    result = _evaluate_scope(
        frame.reset_index(drop=True),
        thresholds,
        sample_weight=sample_weight,
        scope_name="balanced_pooled",
    )
    result["weighting"] = "equal_dataset_contribution"
    return result


def _macro_average_eer(per_dataset_results: Iterable[Dict[str, Any]]) -> float:
    valid_eers = [
        _safe_float(record["threshold_free"].get("eer"))
        for record in per_dataset_results
        if np.isfinite(_safe_float(record["threshold_free"].get("eer")))
    ]
    if not valid_eers:
        return math.nan
    return float(np.mean(valid_eers))


def _select_thresholds(
    validation_frame: pd.DataFrame,
    include_best_f1_threshold: bool = False,
    far_target: float = 0.01,
    threshold_source: str = "validation",
) -> Dict[str, Dict[str, Any]]:
    labels = validation_frame["label"].to_numpy(dtype=int) if not validation_frame.empty else np.asarray([], dtype=int)
    scores = validation_frame["score"].to_numpy(dtype=float) if not validation_frame.empty else np.asarray([], dtype=float)

    eer_selection = _find_threshold_at_eer(labels, scores)
    thresholds: Dict[str, Dict[str, Any]] = {
        "eer_threshold": {
            "threshold": eer_selection["threshold"],
            "source": threshold_source,
            "validation_far": eer_selection["far"],
            "validation_frr": eer_selection["frr"],
            "validation_eer": eer_selection["eer"],
            "validation_f1": math.nan,
            "target_far": math.nan,
            "reason": eer_selection.get("reason"),
        },
    }

    far_selection = find_threshold_at_target_far(labels, scores, target_far=far_target)
    thresholds["far_1pct_threshold"] = {
        "threshold": far_selection["threshold"],
        "source": threshold_source,
        "validation_far": far_selection["far"],
        "validation_frr": far_selection["frr"],
        "validation_eer": far_selection["eer"],
        "validation_f1": math.nan,
        "target_far": far_selection["target_far"],
        "reason": far_selection.get("reason"),
    }

    if include_best_f1_threshold:
        best_f1_selection = _find_best_f1_threshold(labels, scores)
        thresholds["best_f1_threshold"] = {
            "threshold": best_f1_selection["threshold"],
            "source": threshold_source,
            "validation_far": best_f1_selection["far"],
            "validation_frr": best_f1_selection["frr"],
            "validation_eer": math.nan,
            "validation_f1": best_f1_selection["f1"],
            "target_far": math.nan,
            "reason": best_f1_selection.get("reason"),
        }

    return thresholds


def compute_legacy_compatibility_metrics(frame: pd.DataFrame) -> Dict[str, Any]:
    if frame.empty:
        return {
            "min_score": math.nan,
            "max_score": math.nan,
            "threshold": math.nan,
            "eer": math.nan,
            "accuracy": math.nan,
        }

    pred = (frame["score"].to_numpy(dtype=float) >= frame["spoof_score"].to_numpy(dtype=float)).astype(int)
    label = frame["label"].to_numpy(dtype=int)
    accuracy = float((pred == label).mean() * 100.0)
    eer, threshold = compute_eer(label, frame["score"].to_numpy(dtype=float))
    min_score, max_score = _score_range(frame)

    return {
        "min_score": min_score,
        "max_score": max_score,
        "threshold": threshold,
        "eer": eer * 100.0 if np.isfinite(eer) else math.nan,
        "accuracy": accuracy,
    }


def evaluate_binary_classification(
    test_frame: pd.DataFrame,
    validation_frame: Optional[pd.DataFrame] = None,
    include_best_f1_threshold: bool = False,
    far_target: float = 0.01,
    allow_test_threshold_fallback: bool = True,
    fill_policy: Optional[Dict[str, Dict[str, Optional[str]]]] = None,
) -> Dict[str, Any]:
    if validation_frame is not None and not validation_frame.empty:
        effective_validation = validation_frame.reset_index(drop=True)
        threshold_source = "validation"
    elif allow_test_threshold_fallback:
        effective_validation = test_frame.reset_index(drop=True)
        threshold_source = "test_fallback"
    else:
        effective_validation = pd.DataFrame(columns=test_frame.columns)
        threshold_source = "unavailable"

    thresholds = _select_thresholds(
        validation_frame=effective_validation,
        include_best_f1_threshold=include_best_f1_threshold,
        far_target=far_target,
        threshold_source=threshold_source,
    )

    per_dataset = evaluate_per_dataset(test_frame, thresholds, fill_policy=fill_policy)
    raw_pooled = evaluate_pooled(test_frame, thresholds)
    balanced_pooled = evaluate_balanced_pooled(test_frame, thresholds)

    results = {
        "metadata": {
            "label_mapping": {"bonafide": 1, "spoof": 0},
            "score_direction": "higher score means more bonafide",
            "decision_rule": "bonafide if score >= threshold",
            "far_definition": "spoof accepted as bonafide / total spoof",
            "frr_definition": "bonafide rejected as spoof / total bonafide",
            "mdr_definition": "same as FRR for bonafide=1",
            "fill_policy": _normalize_fill_policy(fill_policy),
        },
        "validation": {
            "threshold_source": threshold_source,
            "n_samples": int(len(effective_validation)),
            "n_bonafide": int((effective_validation["label"] == 1).sum()) if not effective_validation.empty else 0,
            "n_spoof": int((effective_validation["label"] == 0).sum()) if not effective_validation.empty else 0,
            "thresholds": thresholds,
        },
        "threshold_free": {
            "per_dataset": per_dataset,
            "macro_average_eer": _macro_average_eer(per_dataset),
            "raw_pooled": raw_pooled,
            "balanced_pooled": balanced_pooled,
        },
        "threshold_based": {
            threshold_name: {
                "per_dataset": [
                    {
                        "dataset_name": record["dataset_name"],
                        "n_bonafide": record["n_bonafide"],
                        "n_spoof": record["n_spoof"],
                        **record["threshold_metrics"][threshold_name],
                    }
                    for record in per_dataset
                ],
                "raw_pooled": raw_pooled["threshold_metrics"].get(threshold_name, {}),
                "balanced_pooled": balanced_pooled["threshold_metrics"].get(threshold_name, {}),
            }
            for threshold_name in thresholds.keys()
        },
    }

    results["summary_text"] = build_readable_summary(results)
    return results


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    header_line = " | ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator_line = "-+-".join("-" * width for width in widths)
    row_lines = [
        " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, separator_line, *row_lines])


def build_readable_summary(results: Dict[str, Any]) -> str:
    sections: List[str] = []

    threshold_rows = []
    for threshold_name, threshold_info in results["validation"]["thresholds"].items():
        threshold_rows.append(
            [
                threshold_name,
                str(threshold_info.get("source", "")),
                _format_float(threshold_info.get("threshold")),
                _format_percent(threshold_info.get("validation_far")),
                _format_percent(threshold_info.get("validation_frr")),
                _format_percent(threshold_info.get("validation_eer")),
                _format_percent(threshold_info.get("validation_f1")),
                _format_percent(threshold_info.get("target_far")),
                threshold_info.get("reason") or "",
            ]
        )
    sections.append(
        "Validation Thresholds\n"
        + _render_table(
            [
                "threshold",
                "source",
                "value",
                "val FAR",
                "val FRR",
                "val EER",
                "val F1",
                "target FAR",
                "note",
            ],
            threshold_rows or [["-", "-", "-", "-", "-", "-", "-", "-", "no thresholds available"]],
        )
    )

    per_dataset_rows = []
    for record in results["threshold_free"]["per_dataset"]:
        per_dataset_rows.append(
            [
                record["dataset_name"],
                str(record["n_bonafide"]),
                str(record["n_spoof"]),
                str(record.get("completion", {}).get("effective_n_bonafide", record["n_bonafide"])),
                str(record.get("completion", {}).get("effective_n_spoof", record["n_spoof"])),
                _format_percent(record["threshold_free"]["eer"]),
                _format_percent(record["threshold_free"]["roc_auc"]),
                _format_percent(record["threshold_free"].get("accuracy")),
                _format_percent(record["threshold_free"].get("bonafide_accuracy")),
                _format_percent(record["threshold_free"].get("spoof_accuracy")),
                record["threshold_free"].get("note") or record["threshold_free"].get("reason") or "",
            ]
        )
    sections.append(
        "Threshold-Free Per-Dataset Metrics\n"
        + _render_table(
            [
                "dataset",
                "n_bonafide",
                "n_spoof",
                "eff_bonafide",
                "eff_spoof",
                "EER",
                "ROC-AUC",
                "accuracy",
                "bonafide_accuracy",
                "spoof_accuracy",
                "note",
            ],
            per_dataset_rows or [["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "no datasets"]],
        )
    )

    pooled_rows = []
    for key in ("raw_pooled", "balanced_pooled"):
        pooled = results["threshold_free"][key]
        pooled_rows.append(
            [
                key,
                str(pooled["n_bonafide"]),
                str(pooled["n_spoof"]),
                _format_percent(pooled["threshold_free"]["eer"]),
                _format_percent(pooled["threshold_free"]["roc_auc"]),
                _format_percent(pooled["threshold_free"].get("accuracy")),
                _format_percent(pooled["threshold_free"].get("bonafide_accuracy")),
                _format_percent(pooled["threshold_free"].get("spoof_accuracy")),
                pooled["threshold_free"].get("reason") or "",
            ]
        )
    pooled_rows.append(
        [
            "macro_average_eer",
            "-",
            "-",
            _format_percent(results["threshold_free"]["macro_average_eer"]),
            "-",
            "-",
            "-",
            "-",
            "",
        ]
    )
    sections.append(
        "Threshold-Free Pooled Metrics\n"
        + _render_table(
            ["scope", "n_bonafide", "n_spoof", "EER", "ROC-AUC", "accuracy", "bonafide_accuracy", "spoof_accuracy", "note"],
            pooled_rows,
        )
    )

    for threshold_name, threshold_result in results["threshold_based"].items():
        threshold_rows = []
        for record in threshold_result["per_dataset"]:
            threshold_rows.append(
                [
                    record["dataset_name"],
                    str(record["n_bonafide"]),
                    str(record["n_spoof"]),
                    _format_float(record.get("threshold")),
                    _format_percent(record.get("accuracy")),
                    _format_percent(record.get("bonafide_accuracy")),
                    _format_percent(record.get("spoof_accuracy")),
                    _format_percent(record.get("precision")),
                    _format_percent(record.get("recall")),
                    _format_percent(record.get("f1")),
                    _format_percent(record.get("far")),
                    _format_percent(record.get("frr")),
                    _format_percent(record.get("mdr")),
                    record.get("reason") or "",
                ]
            )

        raw_pooled = threshold_result.get("raw_pooled", {})
        balanced_pooled = threshold_result.get("balanced_pooled", {})
        threshold_rows.extend(
            [
                [
                    "raw_pooled",
                    "-",
                    "-",
                    _format_float(raw_pooled.get("threshold")),
                    _format_percent(raw_pooled.get("accuracy")),
                    _format_percent(raw_pooled.get("bonafide_accuracy")),
                    _format_percent(raw_pooled.get("spoof_accuracy")),
                    _format_percent(raw_pooled.get("precision")),
                    _format_percent(raw_pooled.get("recall")),
                    _format_percent(raw_pooled.get("f1")),
                    _format_percent(raw_pooled.get("far")),
                    _format_percent(raw_pooled.get("frr")),
                    _format_percent(raw_pooled.get("mdr")),
                    raw_pooled.get("reason") or "",
                ],
                [
                    "balanced_pooled",
                    "-",
                    "-",
                    _format_float(balanced_pooled.get("threshold")),
                    _format_percent(balanced_pooled.get("accuracy")),
                    _format_percent(balanced_pooled.get("bonafide_accuracy")),
                    _format_percent(balanced_pooled.get("spoof_accuracy")),
                    _format_percent(balanced_pooled.get("precision")),
                    _format_percent(balanced_pooled.get("recall")),
                    _format_percent(balanced_pooled.get("f1")),
                    _format_percent(balanced_pooled.get("far")),
                    _format_percent(balanced_pooled.get("frr")),
                    _format_percent(balanced_pooled.get("mdr")),
                    balanced_pooled.get("reason") or "",
                ],
            ]
        )
        sections.append(
            f"Threshold-Based Metrics: {threshold_name}\n"
            + _render_table(
                [
                    "scope",
                    "n_bonafide",
                    "n_spoof",
                    "threshold",
                    "accuracy",
                    "bonafide_accuracy",
                    "spoof_accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "far",
                    "frr",
                    "mdr",
                    "note",
                ],
                threshold_rows or [["-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "no metrics"]],
            )
        )

    far_1pct = results["threshold_based"].get("far_1pct_threshold", {})
    mdr_rows = [
        [
            "raw_pooled",
            _format_percent(far_1pct.get("raw_pooled", {}).get("mdr")),
            _format_percent(far_1pct.get("raw_pooled", {}).get("far")),
            _format_float(far_1pct.get("raw_pooled", {}).get("threshold")),
        ],
        [
            "balanced_pooled",
            _format_percent(far_1pct.get("balanced_pooled", {}).get("mdr")),
            _format_percent(far_1pct.get("balanced_pooled", {}).get("far")),
            _format_float(far_1pct.get("balanced_pooled", {}).get("threshold")),
        ],
    ]
    sections.append(
        "MDR @ FAR=1%\n"
        + _render_table(["scope", "MDR", "achieved FAR", "threshold"], mdr_rows)
    )

    return "\n\n".join(sections)


def to_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [to_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    return value


def dumps_json(data: Dict[str, Any]) -> str:
    return json.dumps(to_json_ready(data), indent=2, sort_keys=False)
