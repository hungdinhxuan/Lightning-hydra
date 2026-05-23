# Auto Experiment + Auto Evaluation Plan

## Goal

- Run multiple YAML configurations to improve model performance.
- Evaluate each run using `summary_results_detailed.txt`.
- Select the best YAML based on a balanced objective:
  - performance close to the teacher on old sets,
  - better than the teacher on the new dataset,
  - strong threshold-free bonafide accuracy on telephony test sets,
  - no major degradation on other sets.

---

## Implementation Checklist

### 1) Prepare experiments

- [ ] Create a separate folder for each config.
- [ ] Use clear YAML names, for example:
  - `conf_01.yaml`
  - `conf_02.yaml`
  - `conf_03.yaml`
- [ ] Change only 1–2 major factors per config to make comparison easier.
- [ ] Record the objective of each config before running it.

---

### 2) Variables to test

- [ ] Replay ratio:
  - `70:30`
  - `60:40`
  - `50:50`
- [ ] Learning rate:
  - lower than baseline
  - same as baseline
  - slightly higher
- [ ] LoRA rank:
  - smaller
  - same
  - larger
- [ ] LoRA dropout:
  - `0.0`
  - `0.05`
  - `0.1`
- [ ] MDT loss weight:
  - baseline
  - slightly higher
- [ ] Distillation weight:
  - off
  - light
  - medium
- [ ] Temperature for distillation:
  - `2`
  - `4`
  - `6`

---

### 3) Train each config

- [ ] Run each YAML one at a time.
- [ ] Save a separate checkpoint for each run.
- [ ] Save full training logs.
- [ ] Use a unique experiment ID for each run.
- [ ] Do not change too many variables at once if the effect is unknown.

---

### 4) Parallel experiment support

- [ ] The server has 4 GPUs available for experiments.
- [ ] Run experiments in parallel when possible to speed up the search.
- [ ] Before launching any job, check which GPUs are currently busy.
- [ ] Do not schedule a new experiment on a GPU that already has an active process.
- [ ] Only use GPUs that are idle or clearly free.
- [ ] Assign at most one heavy experiment job per free GPU unless you have a proven safe multi-job policy.
- [ ] Keep one experiment-to-one-GPU mapping for simplicity and stability.
- [ ] Record which experiment ID is assigned to which GPU.
- [ ] If fewer than 4 GPUs are free, only launch jobs on the currently free GPUs.
- [ ] If all GPUs are busy, wait and retry later instead of forcing a run.
- [ ] Re-check GPU availability before each new training or benchmark launch.
- [ ] Prefer a queue-based launch strategy:
  - detect free GPUs,
  - pop the next config from the queue,
  - assign it to one free GPU,
  - launch,
  - repeat until no free GPU remains.
- [ ] Keep training and evaluation scheduling separate if evaluation is lightweight; otherwise treat evaluation as a GPU job too.
- [ ] Avoid launching evaluation on a GPU already occupied by another training process.
- [ ] Save GPU assignment logs for reproducibility.

---

### 5) Run benchmark

- [ ] After training finishes, benchmark the corresponding checkpoint.
- [ ] Use the standard command exactly as agreed.
- [ ] Replace:
  - `<gpu_number>`
  - `<trained_lora_model>`
  - `<id>`
- [ ] Save outputs in the correct results folder.

---

### 6) Read results

- [ ] Open `summary_results_detailed.txt`.
- [ ] Check metrics per dataset.
- [ ] Prioritize:
  - `1-phone_large-corpus`
  - `telephony_dec25`
  - other old sets
  - the new dataset
- [ ] Clearly note which datasets improved and which dropped.

---

### 7) Evaluate configs

- [ ] Keep a config if:
  - the new dataset improves,
  - old datasets do not drop too much,
  - telephony bonafide accuracy remains strong.
- [ ] Discard a config if:
  - the new dataset improves but old sets collapse,
  - telephony performance drops significantly,
  - results are unstable.
- [ ] Prefer the config with the best trade-off, not just the one with the highest single metric.

---

### 8) Select the best YAML

- [ ] Compare all results.
- [ ] Choose the config with the best balance among:
  - old-set retention,
  - new-set gain,
  - telephony robustness.
- [ ] Record the winning YAML file name.
- [ ] Record why it is better than the other configs.

---

### 9) Final report

- [ ] List all configs that were tried.
- [ ] List the main metric for each config.
- [ ] State the best config clearly.
- [ ] Explain the trade-offs of the best config.
- [ ] State the next step:
  - refine around the best config,
  - or test one new variable in a narrower search space.

---

## Config Selection Rules

- [ ] Do not choose a config based on only one dataset.
- [ ] Do not sacrifice old performance too much just to improve the new dataset.
- [ ] Telephony-related datasets must have high priority.
- [ ] If two configs are almost tied, choose the simpler one.
- [ ] If a more complex config improves only slightly, it is not worth keeping.

---

## Early Skip Rule

- [ ] If `May_08_2026_seonghak_spoof_video_converted` accuracy is below `80%`, stop evaluation for the current config immediately.
- [ ] Do not waste time on the remaining datasets for that config.
- [ ] Jump to the next YAML config immediately.
- [ ] Mark the config as `discard` unless there is a very strong reason to keep it.

---

## Memory Checklist

- [ ] Before starting a new session, read the latest context files.
- [ ] Check the current best YAML config and its results.
- [ ] Check the experiment history and note which configs were kept or discarded.
- [ ] Check the latest benchmark outputs and the current best score.
- [ ] Check whether the early skip rule has already been triggered on `May_08_2026_seonghak_spoof_video_converted`.
- [ ] If continuing from another agent session, first recover the last known state before proposing new configs.
- [ ] Write down the next action clearly so the next session can continue without rereading the full history.
- [ ] Keep the summary short, factual, and updated after every major decision.

---

## Memory Handoff Template

- Current best config: `<yaml_file>`
- Current best checkpoint: `<path>`
- Current best score: `<metric>`
- Last evaluated config: `<yaml_file>`
- Last decision: `<keep/discard/promote>`
- Important failure rule triggered: `<yes/no>`
- Next recommended action: `<next step>`

---

## Outputs to Save

- [ ] YAML file for each experiment.
- [ ] Checkpoint for each run.
- [ ] Benchmark log.
- [ ] `summary_results_detailed.txt`
- [ ] A final summary table including:
  - experiment ID,
  - YAML file,
  - checkpoint path,
  - main metrics,
  - decision: keep / discard / promote.
- [ ] GPU assignment log for each run.

---

## Run Note Template

```text
ID: <id>
Config: <yaml_file>
Checkpoint: <path>
GPU: <gpu_number>
Summary: summary_results_detailed.txt
Decision: keep/discard/promote
Reason: <short reason>
```

---

## Working Principles

- [ ] Test changes in a controlled way.
- [ ] Every run must have a clear objective.
- [ ] Compare against the teacher baseline.
- [ ] Prefer robustness and generalization over a single metric.
- [ ] Once the best config is found, only refine around it.
- [ ] Use parallel execution to improve throughput, but never place new jobs on GPUs that are already busy.

### Early Skip Rule

- [ ] If `May_08_2026_seonghak_spoof_video_converted` accuracy is below `80%`, stop evaluation for the current config immediately.
- [ ] Do not waste time on remaining datasets for that config.
- [ ] Jump to the next YAML config.
- [ ] Record the config as `discard` unless you have a very strong reason to keep it.

### Memory Checklist

- [ ] Before starting a new session, read the latest context files.
- [ ] Check the current best YAML config and its results.
- [ ] Check the experiment history and note which configs were kept or discarded.
- [ ] Check the latest benchmark outputs and the current best score.
- [ ] Check whether the early skip rule already triggered on `May_08_2026_seonghak_spoof_video_converted`.
- [ ] If continuing from another agent session, first recover the last known state before proposing new configs.
- [ ] Write down the next action clearly so the next session can continue without rereading the whole history.
- [ ] Keep the summary short, factual, and updated after every major decision.

### Memory Handoff Template

- Current best config: `<yaml_file>`
- Current best checkpoint: `<path>`
- Current best score: `<metric>`
- Last evaluated config: `<yaml_file>`
- Last decision: `<keep/discard/promote>`
- Important failure rule triggered: `<yes/no>`
- Next recommended action: `<next step>`