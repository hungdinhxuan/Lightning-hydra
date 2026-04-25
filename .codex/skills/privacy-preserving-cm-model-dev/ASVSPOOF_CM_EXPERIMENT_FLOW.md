# ASVspoof CM Privacy Experiment Flow (Local Playbook)

Updated: 2026-03-25
Workspace: `/nvme1/phucdt/wespeaker/examples/asvspoof/cm`

## 1) Goal and Priority

1. Priority-1: recover/approach non-privacy CM performance first using **unfrozen** models.
2. Priority-2: only when CM target hit, run privacy attack (ASR inversion) with LibriSpeech.
3. Hard CM target in this project phase: **ITW EER < 7%** with acceptable LA19.

## 2) Mandatory Trial Pipeline

`screening -> full training -> LA19 eval -> ITW eval -> decision -> (optional) ASR attack -> finalize`

Implemented by: `scripts/run_trial.py`

### Decision notes
- LA19 gate: stop ITW if LA19 degrades beyond `la19_max_rel_degrade`.
- Promote if either:
  - `itw_eer <= decision.target_itw_eer`, or
  - relative ITW improvement >= `promote_itw_rel_improve`.
- If target ITW is reached, ASR attack is triggered immediately.

## 3) Queue/Scheduler Contracts

- Submit by spec only: `bash ./submit_trial.sh --spec <spec.yaml>`
- Scheduler: `scripts/monitor_jobs.py`
- Queue directories:
  - `autonomous/queue/pending`
  - `autonomous/queue/running`
  - `autonomous/queue/deferred`
  - `autonomous/queue/done`
- Priority sort rule: `(priority ASC, queued_time ASC)`

Queue hygiene for fast iteration:
- Keep only top-priority hypotheses in `pending`.
- Move stale/duplicated trials to `deferred`.
- Do not touch files in `running` unless trial is confirmed stale/dead.

## 4) Spec Authoring Pattern

Use one-factor ablations and prefer override-driven specs.

### New override capability
`run_trial.py` supports:
- `model.overrides` (deep-merge into base config)
- `pipeline.screening.config_overrides`
- `pipeline.full_training.config_overrides`

This enables fast experimentation without duplicating many base YAML configs.

### Recommended baseline references
- Full unfrozen: `conf/tcm_add_xlsr2_300m_unfrozen_tuned_v2_fullstrength.yaml`
- L1-L7 unfrozen: `conf/tcm_add_xlsr2_300m_attentive_agg_l1_l7_unfrozen_tuned_v2.yaml`

## 5) Current Best Practice (March 22, 2026)

1. Run unfrozen full-layer and unfrozen L1-L7 in parallel (one per MIG).
2. Test one axis at a time:
   - LR/schedule stability
   - batch size scaling for 40GB MIG
   - temporal context (`num_frms`)
   - attentive-agg sharpness (`attn_temperature`, hidden size)
   - layer set changes (`agg_layers`)
3. Trigger privacy attack only when CM target is met.

## 6) Privacy Attack Trigger (LibriSpeech)

When promoted/target-hit:
- Prepare LibriSpeech ASR data if missing via `local/prepare_librispeech_asr.sh`
- Train attackers with CM frontend features:
  - `bilstm_hybrid`
  - `conformer`

## 6.1) Speaker Privacy Attack Trigger (LA19)

When a CM trial is promoted:

1. Train speaker attacker with:
   - frozen CM frontend features
   - LA19 train `protocol.txt` speaker labels
   - attacker backbone: ECAPA-TDNN (`ECAPA_TDNN_GLOB_c512`, `ASTP`)
2. Evaluate on LA19 eval using cosine similarity pairs from attacker embeddings.
3. Store artifacts in trial-specific directory:
   - `train_log.json`
   - `summary.json`
   - `best.pt`
4. Decision gate:
   - if attacker-EER is low => speaker leakage exists => model needs mitigation before promotion
   - if attacker-EER is high with good CM metrics => candidate is privacy-strong

## 6.2) Dual-MIG Execution Plan (New)

For full speaker-privacy attack on promoted CM:

1. Launch 2 parallel runs on:
   - `MIG-8cdeef83-092c-5a8d-a748-452f299e1df0`
   - `MIG-6e4275af-2db0-51f1-a601-7ad8a1002745`
2. Start with:
   - run-A `batch_size=96`
   - run-B `batch_size=128`
3. Keep the better stable throughput config as default for next attacks.

### Recommended launch command

Use:

- `local/run_speaker_privacy_attack.sh <config> <output_dir> <MIG_UUID>`

Run via detached session (`tmux`) for long jobs.

## 6.3) Pre-run Storage Checklist (New)

Before any full privacy attack:

1. Ensure free space > 500GB.
2. Prune old failed/rejected trial artifacts:
   - delete `cm_exp_full/`, `cm_exp_screen/`
3. Keep metadata and metrics only.

Reason: avoid `Errno 28 No space left on device` mid-training.

## 7) Operational Checklist per Iteration

1. Read memory files: `current_status.md`, `handoff.md`, `experiment_table.csv`, `registry.jsonl`.
2. Check event folders: `RESULT_READY`, `FAILED`, `NEED_DECISION`.
3. Prepare 2-6 next candidate specs (single-factor changes).
4. Submit specs and verify pending order.
5. Keep queue clean (defer duplicates/low-value leftovers).
6. Update `memory/handoff.md` with active focus and next candidates.

## 8) Speaker-Privacy Dry-Run Checklist

1. Run dry-run config with small subsets and 2 epochs.
2. Verify files are produced:
   - `summary.json`, `train_log.json`, `best.pt`
3. Verify metadata consistency:
   - train/eval speaker counts
   - pair generation stats (pos/neg)
4. If dry-run passes, launch full speaker attack for promoted CM trial.

## 9) CM/SASV Runtime Control (Practical)

### 9.1 Single-monitor invariant

Always keep exactly one monitor:

```bash
ps -eo pid,cmd | rg "examples/asvspoof/cm/scripts/monitor_jobs.py" | rg -v rg
```

If more than one exists: kill all and relaunch one.

### 9.2 Fast relaunch for dead queue items

1. Put queue file back to `queue/pending`.
2. Set trial `status.json` back to `PENDING`.
3. Run:
```bash
python scripts/monitor_jobs.py --once
```
4. Confirm process exists and log advances.

### 9.3 Batch fallback ladder

When unstable/OOM:

1. `target=48, max=56` (default)
2. `target=40, max=48` (safe fallback for heavy arcsub)
3. Keep `eval_batch_size >= 64`

### 9.4 Dashboard refresh cadence

Update every 15 minutes (cron) using:

```bash
python examples/asvspoof/cm/scripts/update_memory_docs.py
```

Required dashboard sections:

1. CM training
2. ASR attack on CM
3. ASR attack on ASV
4. ASV attack on CM
5. SASV joint finetuning