# Evaluation Metrics And Fill Policy

This folder stores evaluation-time config for benchmark reporting.

Main use now:
- define per-dataset missing-class completion policy
- explain which metrics are threshold-free vs threshold-based
- keep benchmark summary meaning consistent across mixed datasets

Example config:
- [April_2026_benchmark.yaml](/nvme2/hungdx/Lightning-hydra/configs/eval/April_2026_benchmark.yaml)

## How Benchmark Picks Eval Config

Two ways:

### 1. Explicit flag

Pass config directly:

```bash
uv run ./scripts/benchmark_py/benchmark.py \
  -g 1 \
  -c your/config \
  -b data/April_2026_benchmark \
  -m /path/to/model.ckpt \
  -r logs/results/April_2026_benchmark \
  -n run_name \
  --eval-config configs/eval/April_2026_benchmark.yaml
```

### 2. Auto-detect fallback

If `--eval-config` is omitted, benchmark tries:

```text
configs/eval/<benchmark_folder_name>.yaml
```

Example:
- `-b data/April_2026_benchmark`
- auto-detects `configs/eval/April_2026_benchmark.yaml`

## Score Convention

Assumptions used by evaluator:
- `label=1` means `bonafide`
- `label=0` means `spoof`
- higher score means more `bonafide`
- decision rule at threshold `t`:
  - predict `bonafide` if `score >= t`
  - predict `spoof` if `score < t`

## Metric Groups

### Threshold-Free Metrics

These do not need one fixed operating threshold on test set.

#### 1. Per-dataset EER

Equal Error Rate for one dataset.

- sweep threshold over score range
- compute:
  - `FAR = spoof accepted as bonafide / total spoof`
  - `FRR = bonafide rejected as spoof / total bonafide`
- `EER` is operating point where `FAR` and `FRR` are equal, or closest by interpolation

Interpretation:
- lower is better
- `0%` is perfect

#### 2. Macro-average EER

Unweighted mean of valid per-dataset EER values.

Formula:
```text
macro_EER = mean(EER_d for all datasets d with valid EER)
```

Notes:
- each dataset contributes equally
- dataset size does not matter
- datasets with undefined EER are skipped from mean

#### 3. Raw pooled EER

Concatenate all samples from all datasets, then compute one EER.

Formula:
```text
raw_pooled_EER = EER(concat(all datasets))
```

Notes:
- larger datasets dominate
- good when you want sample-level overall performance

#### 4. Balanced pooled EER

Compute pooled EER with equal dataset contribution.

Implementation:
- keep all samples
- assign sample weights so each dataset has same total weight
- compute weighted ROC/EER

Notes:
- avoids huge dataset dominating pooled result
- closer to "dataset-balanced" overall score

#### 5. ROC-AUC

Area under ROC curve.

- threshold-free ranking metric
- probability model ranks bonafide above spoof

Interpretation:
- higher is better
- `100%` perfect separation
- `50%` random ranking

## Threshold-Based Metrics

These need fixed threshold chosen on validation data, then applied to test data.

Important rule:
- threshold must be selected on validation
- test set only used for final measurement
- no threshold tuning on test

Supported threshold choices:
- threshold at validation EER
- threshold at validation `FAR=1%`
- optional best-F1 threshold on validation

### 1. Accuracy

Formula:
```text
accuracy = correct predictions / total samples
```

### 2. Precision

For positive class = bonafide.

Formula:
```text
precision = true bonafide accepted / all bonafide predictions
```

### 3. Recall

For positive class = bonafide.

Formula:
```text
recall = true bonafide accepted / total bonafide
```

### 4. F1

Formula:
```text
F1 = 2 * precision * recall / (precision + recall)
```

### 5. FAR

False Acceptance Rate.

Formula:
```text
FAR = spoof accepted as bonafide / total spoof
```

Lower is better.

### 6. FRR

False Rejection Rate.

Formula:
```text
FRR = bonafide rejected as spoof / total bonafide
```

Lower is better.

### 7. MDR

Miss Detection Rate.

In this binary setup:
```text
MDR = FRR
```

Reason:
- "miss" means missing bonafide acceptance
- same event as bonafide rejected as spoof

### 8. MDR @ FAR=1%

Procedure:
1. choose threshold on validation so `FAR ~= 1%`
2. apply that threshold to test
3. report test `MDR`

Use:
- shows bonafide miss rate under strict spoof-acceptance control

## Missing-Class Completion

Problem:
- some datasets contain only spoof or only bonafide
- direct per-dataset EER/AUC is undefined because ROC needs both classes

Solution:
- fill missing class from another dataset, based on YAML config

Config format:
```yaml
fill_policy:
  target_dataset:
    bonafide_source: some_dataset_or_null
    spoof_source: some_dataset_or_null
```

Meaning:
- if target dataset has no bonafide, evaluator may borrow bonafide samples from `bonafide_source`
- if target dataset has no spoof, evaluator may borrow spoof samples from `spoof_source`

Important scope:
- fill applies only to threshold-free per-dataset EER/AUC
- threshold-based per-dataset metrics still use raw target dataset samples
- pooled metrics already see all datasets together, so they usually do not need fill

Why this split:
- threshold-free EER/AUC need both classes to be mathematically defined
- threshold-based metrics should still reflect true target-dataset operating behavior under chosen global threshold

### When Result Is Still `NaN`

Per-dataset EER/AUC stays `NaN` if:
- dataset has only one class
- and no fill source is configured for missing class
- or configured source exists but also lacks needed class

Example:
- if `M-AILABS` has only bonafide
- and `spoof_source: null`
- then per-dataset EER/AUC remain undefined

## Summary Files

Benchmark final compact summary now shows:

```text
Dataset | EER | ROC-AUC | Accuracy
```

Meaning:
- `EER`: threshold-free per-dataset EER, with fill-policy applied if needed
- `ROC-AUC`: threshold-free per-dataset AUC, with fill-policy applied if needed
- `Accuracy`: threshold-based accuracy from raw dataset predictions

Detailed file also shows:
- original class counts: `n_bonafide`, `n_spoof`
- effective counts after fill: `eff_bonafide`, `eff_spoof`
- fill note, for example:
  - `filled bonafide from M-AILABS (+54998)`

## Current April 2026 Config

Current file:
- [April_2026_benchmark.yaml](/nvme2/hungdx/Lightning-hydra/configs/eval/April_2026_benchmark.yaml)

Current intent:
- `2026_April_Dataset_jiwon`, `MLAAD_v7`, `MLAAD_v8`, `MLAAD_v9`, `MLAAD_dev_v10_April21_2026`
  borrow bonafide from `M-AILABS`
- `M-AILABS` itself has no spoof fill source, so its per-dataset EER/AUC can remain `NaN`

## Code Pointers

Main implementation:
- [scripts/benchmark_py/binary_eval.py](/nvme2/hungdx/Lightning-hydra/scripts/benchmark_py/binary_eval.py:43)
- [scripts/benchmark_py/execution.py](/nvme2/hungdx/Lightning-hydra/scripts/benchmark_py/execution.py:195)
- [scripts/benchmark_py/eer.py](/nvme2/hungdx/Lightning-hydra/scripts/benchmark_py/eer.py:1)
- [scripts/calculate_pooled_eer.py](/nvme2/hungdx/Lightning-hydra/scripts/calculate_pooled_eer.py:1)

Synthetic validation:
- [tests/test_binary_eval.py](/nvme2/hungdx/Lightning-hydra/tests/test_binary_eval.py:97)
