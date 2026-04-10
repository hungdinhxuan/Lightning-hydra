# 🧠 Autonomous Execution Guide

## Privacy-Preserving CM Research System

---

# 1. Purpose

This document defines **how you (the AI agent)** must operate within the experiment system.

You are **NOT responsible for running training manually**.

Instead, your role is:

1. Plan experiments
2. Submit trials
3. Read results
4. Decide next steps

All heavy computation is handled by the **background scheduler system**.

---

# 2. System Overview

The system consists of:

### 1. Planner (YOU)

* Reads memory
* Creates trial spec
* Submits job
* Makes decisions

### 2. Scheduler (background)

* Assigns jobs to MIG GPUs
* Starts workers

### 3. Worker

* Runs:

  * small screening
  * full training
  * evaluation
  * privacy test

### 4. Memory

* Stores all experiment history

### 5. Event System

* Notifies when:

  * trial finished
  * trial failed
  * decision needed

---

# 3. Your Responsibilities

You must ALWAYS follow this loop:

---

## Step 1 — Read Memory

Before doing anything, read:

* `memory/current_status.md`
* `memory/handoff.md`
* `memory/experiment_table.csv`
* `memory/registry.jsonl`

Goal:

* understand current progress
* identify best model
* identify failures

---

## Step 2 — Detect Events

Check:

```
events/RESULT_READY/
events/FAILED/
events/NEED_DECISION/
events/PRIVACY_READY/
```

---

### If RESULT_READY:

* load metrics
* compare with baselines
* decide: reject / keep / promote

---

### If FAILED:

* read error log
* determine cause:

  * OOM
  * divergence
  * config bug
* plan fix trial

---

### If NEED_DECISION:

* choose next experiment direction

### If PRIVACY_READY:

* run both privacy attacks for promoted CM:
  * ASR attack (linguistic privacy)
  * speaker attack (speaker privacy)
* compare against previous promoted model
* write final privacy verdict before next CM iteration
* enforce disk check and dual-MIG launch policy for speaker attack

---

## Step 3 — Decide Next Trial

Use search policy:

* improve best model
* test one new component
* avoid stacking unverified changes

---

## Step 4 — Create Trial Spec

Write a YAML spec:

* model config
* hypothesis
* pipeline steps
* trial type:

  * screening
  * full_privacy

---

## Step 5 — Submit Trial

Run:

```bash
python scripts/submit_trial.py --spec <path_to_yaml>
```

For speaker-privacy attack dry-run/full-run, use local runner scripts under:

* `examples/asvspoof/cm/local/run_speaker_privacy_attack_dryrun.sh`
* `examples/asvspoof/cm/local/run_speaker_privacy_attack.sh`

Operational notes:

* Use absolute paths for config/output when invoking runner scripts.
* Prefer detached sessions (`tmux`) for long runs.
* Always capture and report:
  - best attacker-EER
  - leak level (`high|medium|low`)
  - run stability (OOM/EXIT reason if any)

---

## Step 6 — Exit

DO NOT:

* wait for training
* monitor GPU
* run training manually

The system will wake you later.

---

# 4. Trial Lifecycle

Each trial follows:

```
PENDING
→ RUNNING_SMALL
→ RUNNING_FULL
→ EVAL_LA19
→ EVAL_ITW
→ (optional) ASR_ATTACK
→ FINALIZED
```

---

# 5. Experiment Loop (MANDATORY)

Every experiment must follow:

---

## 1. Small Screening

* subset data
* quick epochs

Goal:

* detect instability early

---

## 2. Full Training

Only if screening passes.

---

## 3. LA19 Evaluation

If bad → STOP

---

## 4. ITW Evaluation

Only if LA19 acceptable

---

## 5. Decision

* Reject
* Keep
* Promote

---

## 6. Privacy Evaluation

Only for promoted models.

---

## 7. Repeat

---

# 6. Decision Rules

## Reject if:

* worse than L1-L7 baseline
* unstable training

## Keep if:

* better than L1-L7

## Promote if:

* strong ITW improvement
* stable
* good candidate for privacy

---

# 7. Baselines (ALWAYS COMPARE)

You must compare against:

* `TCM_ADD` (no privacy)
* `L1-L7` baseline
* SafeEar

---

# 8. GPU System (IMPORTANT)

You have 2 MIG GPUs:

* MIG-0 → CM training
* MIG-1 → evaluation / ASR attacker

Rules:

* NEVER assign jobs manually
* Scheduler handles GPU assignment

---

# 9. External Memory (CRITICAL)

These files are your memory:

### registry.jsonl

Full history

### experiment_table.csv

All results

### current_status.md

Current running jobs

### handoff.md

Key insights + next steps

---

## Rule:

You must ALWAYS read memory before planning.

---

# 10. Event Handling

## RESULT_READY

→ analyze results
→ update decision
→ create next trial

## FAILED

→ debug cause
→ create fix trial

## NEED_DECISION

→ plan next experiment

---

# 11. Forbidden Actions

DO NOT:

* run training manually
* modify running jobs
* delete memory files
* skip experiment loop

---

# 12. Success Condition

Stop ONLY if:

* target performance reached OR
* compute budget exhausted

Otherwise:
→ continue experiments

---

# 13. Core Research Goal

Find a model that:

* beats L1-L7 baseline
* improves ITW
* preserves LA19 performance
* reduces linguistic leakage

---

# 14. Key Strategy Reminder

Focus on:

1. Better XLS-R fusion
2. Better pooling
3. Lightweight temporal modeling
4. Then privacy mechanisms

---

# 15. Operations SOP (Production)

## 15.1 Startup checklist

1. Verify only one monitor is active:
```bash
ps -eo pid,cmd | rg "examples/asvspoof/cm/scripts/monitor_jobs.py" | rg -v rg
```
2. If none, start monitor.
3. Run one `--once` tick and verify workers are spawned.

## 15.2 Health checks (every loop)

1. Process check:
```bash
ps -eo pid,cmd | rg "run_trial.py|wespeaker/bin/train.py|monitor_jobs.py" | rg -v rg
```
2. Queue check:
```bash
ls -1 autonomous/queue/pending
ls -1 autonomous/queue/running
```
3. Status check:
```bash
cat autonomous/trials/<trial_id>/status.json
```
4. Log check:
```bash
tail -n 100 autonomous/trials/<trial_id>/logs/worker.log
tail -n 100 autonomous/trials/<trial_id>/cm_exp_screen/train.log
```

## 15.3 Dead-task recovery (exact order)

1. Stop duplicate/stale monitor processes.
2. Stop stale worker process for dead trial.
3. Move trial queue YAML back to `pending` if misplaced.
4. Reset trial `status.json` to `PENDING` if needed.
5. Rebuild `running_slots.json` to reflect only real alive workers.
6. Relaunch single monitor.
7. Execute one `monitor_jobs.py --once`.
8. Confirm trial is relaunched and train log advances.

## 15.4 Dashboard update workflow

After every recovery or major queue change:
```bash
python examples/asvspoof/cm/scripts/update_memory_docs.py
```

Ensure dashboard shows these sections:
- CM training (result + status)
- ASR attack on CM
- ASR attack on ASV
- ASV attack on CM
- SASV joint finetuning

---

# END