---
name: "privacy-preserving-cm-model-dev"
description: "Develop privacy-preserving CM models with XLS-R. Focus on outperforming L1-L7 baseline and SafeEar while minimizing linguistic leakage. Fully autonomous execution with background runners."
---

# 🧠 AI Research Agent Task Specification

## Project: Privacy-Preserving CM (Autonomous Research System)

---

# 1. Mission Objective

Design CM models that:

1. Achieve strong performance on:
   - LA19 eval
   - ITW

2. Reduce linguistic leakage:
   - Measured via ASR attacker (WER/CER)

3. Outperform:
   - `TCM_ADD` (no-privacy baseline)
   - `L1-L7 frontend` baseline

4. Compete with:
   - SafeEar

5. Preserve speaker privacy:
   - Front-end features should not support strong speaker recognition.
   - Measured by a speaker-privacy attacker on ASVspoof LA19.

---

# 2. 🎯 Target Performance (MANDATORY)

## CM Targets

| Metric | Target |
|------|--------|
| LA19 EER | ≤ `TCM_ADD` |
| ITW EER | ≥ 10% relative improvement vs L1-L7 |
| Stability | Must converge |

## Privacy Targets

| Metric | Target |
|------|--------|
| ASR WER | ≥ +20% relative |
| ASR CER | ≥ +15% |
| Speaker attack EER (LA19 eval) | Higher is better privacy |

---

## Success Definition

A model is valid only if:

- ITW improves
- LA19 not degraded significantly
- ASR attacker fails more
- Speaker attacker does not become stronger (no EER drop vs baseline CM)

---

# 3. 🔁 Detailed Experiment Loop (CRITICAL)

Every trial MUST follow this pipeline:

---

## Step 0 — Workspace Validation

- Resolve paths:
  - REPO_ROOT, CM_RECIPE_DIR, datasets
- Abort if invalid

---

## Step 1 — Small-scale Screening

Run fast test:

- 5–10% dataset
- 3–5 epochs

Goal:
- check training stability
- check signal vs baseline

---

## Step 2 — Full Training

Only if Step 1 passes:

- Full dataset
- Proper schedule
- Save best checkpoint

---

## Step 3 — CM Evaluation

- Evaluate LA19 eval
- If LA19 not acceptable → STOP ITW

- If acceptable:
  - evaluate ITW

---

## Step 4 — Decision

### Case A — Worse than baseline
→ Reject  
→ Log cause  

### Case B — Slight improvement
→ Tune:
- LR
- pooling
- fusion weights

### Case C — Strong improvement
→ Promote to privacy stage

---

## Step 5 — Privacy Evaluation

Only for promoted models:

- Run ASR attackers:
  - BiLSTM
  - Conformer
- Run Speaker attacker:
  - CM frontend (frozen) + ECAPA backend classifier (train on LA19 train speaker IDs from `protocol.txt`)
  - Evaluate speaker verification EER on LA19 eval from cosine pairs of attacker embeddings

### Speaker Privacy Protocol (Expanded)

For each promoted CM trial (especially `cmcand_full_unf_tunedv2_20260322_144839`):

1. Build attacker input from CM frontend features only (frontend frozen).
2. Train speaker attacker with LA19 train speaker labels:
   - baseline attacker: `ECAPA_TDNN_GLOB_c512` + `ASTP`
3. Evaluate on LA19 eval with speaker verification pairs from attacker embeddings.
4. Report:
   - speaker-attack EER (primary)
   - train accuracy (secondary)
   - pair stats (pos/neg counts, unique speakers)
5. Interpret:
   - lower attacker-EER => stronger speaker leakage
   - higher attacker-EER => better speaker privacy

### Required Speaker-Privacy Outputs

- `train_log.json`
- `summary.json`
- best checkpoint path
- short conclusion line in memory update:
  - `speaker_leakage=high|medium|low`
  - `next_action=<mitigation>`

### Mitigation Tracks (for next CM iterations)

If speaker leakage is high, prioritize one-factor changes:

1. feature bottleneck after frontend projection
2. speaker-adversarial head (GRL) during CM training
3. layer-drop / random layer mixing in frontend aggregation
4. stronger temporal content pooling with reduced speaker cues

---

## Step 6 — Update Memory

- registry.jsonl
- experiment_table.csv
- handoff.md

---

## Step 7 — Next Trial Planning

Choose:

- refine best model
- test fusion variant
- add privacy module

---

## 🔁 Repeat Until:

- Target met OR
- Budget exhausted

---

# 4. Baseline System (MANDATORY MATRIX)

Must run:

1. TCM_ADD full-layer + frozen
2. TCM_ADD full-layer + unfrozen
3. TCM_ADD L1-L7 + frozen
4. TCM_ADD L1-L7 + unfrozen

---

# 5. Model Exploration Priority

## Priority 1 (CRITICAL)
Beat L1-L7:

- weighted sum
- grouped fusion
- layer gating

## Priority 2
Pooling:

- attentive stats pooling

## Priority 3
Temporal:

- Conv1D

## Priority 4
Privacy:

- bottleneck
- content adversarial

---

# 6. 🧠 Search Policy

- Change ONE factor per trial
- Only combine validated components
- Avoid stacking unverified modules

---

# 7. ⚙️ Execution System

## Agent Behavior

Agent should use the autonomous queue first. Manual direct launch is only for recovery/debug.

Instead:

1. Create spec.yaml
2. Submit job
3. Ensure scheduler/monitor is healthy
4. Update dashboard docs

---

# 7.1 🛠️ Operational Runbook (CM + SASV)

## 8.1 Core files

- Scheduler: `examples/asvspoof/cm/scripts/monitor_jobs.py`
- Worker: `examples/asvspoof/cm/scripts/run_trial.py`
- Dashboard updater: `examples/asvspoof/cm/scripts/update_memory_docs.py`
- Queue:
  - `examples/asvspoof/cm/autonomous/queue/pending`
  - `examples/asvspoof/cm/autonomous/queue/running`
  - `examples/asvspoof/cm/autonomous/queue/deferred`
- Slots: `examples/asvspoof/cm/autonomous/queue/running_slots.json`

## 8.2 Canonical commands

Start scheduler:
```bash
nohup /nvme1/phucdt/miniconda3/envs/wespeaker/bin/python \
  /nvme1/phucdt/wespeaker/examples/asvspoof/cm/scripts/monitor_jobs.py \
  --poll-sec 20 \
  > /nvme1/phucdt/wespeaker/examples/asvspoof/cm/autonomous/monitor.log 2>&1 &
```

One scheduling tick (safe):
```bash
/nvme1/phucdt/miniconda3/envs/wespeaker/bin/python \
  /nvme1/phucdt/wespeaker/examples/asvspoof/cm/scripts/monitor_jobs.py --once
```

Update docs/dashboard:
```bash
/nvme1/phucdt/miniconda3/envs/wespeaker/bin/python \
  /nvme1/phucdt/wespeaker/examples/asvspoof/cm/scripts/update_memory_docs.py
```

## 8.3 Required monitor policy

- Only **one** `monitor_jobs.py` process may run at a time.
- If multiple monitors are found, stop all and relaunch one clean instance.
- Never leave stale `running_slots.json` after abrupt worker death.

## 8.4 Recovery policy (when tasks "die")

1. Check `status.json`, `logs/worker.log`, `cm_exp_screen/train.log`.
2. Verify queue consistency (`pending`/`running` file location must match status).
3. Rebuild `running_slots.json` to match real live workers.
4. Requeue failed job to `pending`.
5. Run one scheduler tick.
6. Re-check process + log progress before declaring healthy.

## 8.5 Batch-size policy (current)

- Default screening target batch: 48
- Default screening max batch: 56
- Default full-training target batch: 48
- For unstable/OOM arcsub SASV: reduce to screening target 40, max 48

---

# 8. 🆕 Speaker Privacy Track v2 (Operational)

This direction is now mandatory for promoted CM candidates, starting from:

- `cmcand_full_unf_tunedv2_20260322_144839`

## Objectives

1. Verify whether CM frontend leaks speaker identity.
2. Quantify leakage with attacker-EER on LA19 eval.
3. Run resource-optimized attacks on dual 48GB MIGs.

## Runtime Policy

1. Always run two speaker-attacker jobs in parallel (one per MIG) when both are available.
2. Use high-but-stable batch sizes to maximize throughput:
   - MIG-A: start `batch_size=96`
   - MIG-B: start `batch_size=128`
3. If OOM occurs, reduce only the failing job by one step (`-16` batch).

## Disk Hygiene Policy (before long runs)

1. Prune heavy artifacts from failed/rejected trials:
   - remove `cm_exp_full/` and `cm_exp_screen/`
   - keep `status.json`, `spec.yaml`, `metrics.json`, `scores/`
2. Remove stale dry-run outputs and oversized transient logs.
3. Abort new full run if free disk is below 500GB.

## Acceptance Readout

For each speaker-attacker run, report:

- best attacker-EER
- epoch at best EER
- pair stats (`num_pos_pairs`, `num_neg_pairs`, `num_speakers`)
- leak level:
  - `high` if EER < 15
  - `medium` if 15 <= EER < 30
  - `low` if EER >= 30

Lower EER means stronger speaker leakage.

## Required Scripts

### submit_trial.sh
- create trial
- enqueue job

### run_trial.py
Pipeline:
- small test
- full training
- CM eval
- privacy eval (optional)
- finalize

### monitor_jobs.py
- schedule GPU
- detect failure
- emit events

### finalize_trial.py
- compute metrics
- write report
- update memory

---

# 9. 📢 Event Trigger System

Agent wakes ONLY when:

## RESULT_READY
→ trial finished

## FAILED
→ error occurred

## NEED_DECISION
→ planning required

---

# 10. Speaker Privacy Attack Protocol (MANDATORY)

## Goal

Given CM model `(config + checkpoint)`, test whether CM frontend features leak speaker information.

## Data labels

- Use speaker ID from `protocol.txt` column 1 (`LA_xxxx`), not `utt2spk` (bonafide/spoof).
- Train set: `data/asvspoof2019_LA_train`
- Eval set: `data/asvspoof2019_LA_eval`

## Baseline attacker

- Frontend: CM frontend (frozen)
- Backend: `ECAPA_TDNN_GLOB_c512`
- Training target: speaker classification on LA19 train speakers
- Eval metric: speaker verification EER on LA19 eval (cosine scores from attacker embeddings)

## Core scripts

- Train/eval attacker:
  - `examples/asvspoof/cm/local/train_privacy_speaker_attack_cm_frontend.py`
- Config:
  - `examples/asvspoof/cm/conf/privacy_speaker_attack_ecapa_cm_frontend.yaml`
- Fast dry-run:
  - `examples/asvspoof/cm/local/run_speaker_privacy_attack_dryrun.sh`

## Standard run command

```bash
CUDA_VISIBLE_DEVICES=<MIG_UUID> \
PYTHONPATH=/nvme1/phucdt/wespeaker/examples/asvspoof/cm:/nvme1/phucdt/wespeaker \
/nvme1/phucdt/miniconda3/envs/wespeaker/bin/python \
examples/asvspoof/cm/local/train_privacy_speaker_attack_cm_frontend.py \
  --config examples/asvspoof/cm/conf/privacy_speaker_attack_ecapa_cm_frontend.yaml
```

## Dry-run command

```bash
examples/asvspoof/cm/local/run_speaker_privacy_attack_dryrun.sh \
  /nvme1/phucdt/wespeaker/examples/asvspoof/cm/conf/privacy_speaker_attack_ecapa_cm_frontend_dryrun.yaml \
  /nvme1/phucdt/wespeaker/examples/asvspoof/cm/exp/privacy_speaker_attack_ecapa_cm_frontend_dryrun \
  MIG-8cdeef83-092c-5a8d-a748-452f299e1df0
```

## Interpretation

- Lower speaker-EER => attacker is stronger => speaker leakage is higher.
- Higher speaker-EER => stronger speaker privacy.
- Compare absolute EER and deltas across CM variants (frozen/unfrozen, full/l1l7, mlfg/sls, etc.).

---

# 11. Speaker-Privacy Improvement Directions

Prioritize preserving CM performance first, then tighten privacy with controlled ablations:

1. Frontend disentanglement
- Add speaker-adversarial branch (GRL) on frontend embeddings.
- Increase adversarial weight gradually after CM converges.

2. Information bottleneck
- Add low-rank/channel bottleneck before backend CM head.
- Enforce stochastic/noise bottleneck only during train.

3. Speaker confusion regularization
- Minimize speaker discriminability on train speakers (uniform/confusion loss).
- Keep spoof/bonafide discrimination unchanged.

4. Temporal/content-preserving masking
- Random temporal/channel masking targeted to speaker traits (pitch/formant-heavy channels).
- Pair with consistency loss to preserve CM decisions.

5. Multi-objective scheduling
- Phase 1: match no-privacy CM target.
- Phase 2: turn on speaker privacy loss with small coefficient.
- Phase 3: sweep privacy weight and select Pareto-optimal checkpoints.

---

# 12. Local Execution Playbook

For this workspace implementation and the latest queue/decision conventions, use:

- `ASVSPOOF_CM_EXPERIMENT_FLOW.md`

# 13. 🧠 External Memory System

## Files

### registry.jsonl
- full history (append-only)

### experiment_table.csv
- structured results

### current_status.md
- running jobs

### handoff.md
- reasoning context

---

## Required Behavior

Agent must:

1. Read all memory before planning
2. Update memory after every trial

---

# 14. 🖥️ MIG GPU Policy (DETAILED)

Use these MIG devices by default for CM/privacy attacks:
  GPU 0: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (UUID: GPU-c07cf4f2-fe74-ec31-7035-f01beb777c12)
    MIG 2g.48gb Device 0 (UUID: MIG-8cdeef83-092c-5a8d-a748-452f299e1df0)
    MIG 2g.48gb Device 1 (UUID: MIG-6e4275af-2db0-51f1-a601-7ad8a1002745)

Scheduling rule:
  - one long-running job per MIG
  - prefer CM training/eval on one MIG and privacy attacker on the other when both queues are active

---
---

## Parallel Pipeline

Example:

---

## Rules

- 1 job per MIG
- No overcommit

---

## OOM Handling

If OOM:

1. Reduce batch size
2. Reduce segment length
3. Use grad accumulation

---

# 15. Logging Requirements

Each trial MUST produce:

- metrics.json
- status.json
- report.md
- config.yaml
- final checkpoint

---

# 16. Privacy Evaluation Ladder

- Trial 1–2: proxy allowed
- Trial ≥3: full ASR required
- Final models: both attackers required

---

# 17. Decision Rules

## Reject
- worse than L1-L7
- unstable

## Keep
- better than L1-L7

## Promote
- strong ITW + privacy gain

---

# 18. Failure Handling

On failure:

1. Retry once
2. Fix:
   - OOM → reduce batch
   - divergence → lower LR
3. Log root cause

---

# 19. 🔁 Autonomous Research Loop

After each trial:

1. Compare all results
2. Identify bottleneck:
   - ITW weak
   - LA19 weak
   - privacy weak
3. Plan next trial

---

## Stop Conditions

Stop ONLY if:

- target met OR
- budget exhausted

Else → continue

---

# 20. SafeEar Baseline

Must compare against:

- LA19 EER
- ITW EER
- ASR WER/CER

---

# 21. Primary Research Question

Can we beat L1-L7 by:

- better fusion
- better pooling
- privacy-aware design

---

# END