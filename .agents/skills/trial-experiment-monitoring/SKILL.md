---
name: trial-experiment-monitoring
description: Runs full-strength Lightning-Hydra training with mandatory wandb and Hydra overrides, launches jobs via nohup with PID tracking (or tmux per run_background_task for detachable sessions), schedules single-GPU execution with MIG-aware CUDA_VISIBLE_DEVICES, drives queue and cron-triggered monitor_jobs.py, installs crontab via scripts/install_trial_monitor_cron.sh and removes it via scripts/remove_trial_monitor_cron.sh once the experiment is finished, results are ready, and the human has completed plan review. Wakes the agent on FAILED/NEED_DECISION, and also performs a two-stage post-train workflow: on RESULT_READY it triggers “evaluate model”, then on evaluation completion it triggers “create summary report CSV”, using event lines in events.jsonl as the handoff. Aggregates benchmark metrics to CSV and finalizes. Use when running or resuming long trials, automating monitor cron, or debugging failed full-strength jobs in this repo—not for initial external-repo porting (see external-repo-integration).
---

# Trial experiment and monitoring

## Purpose

Use for **production training**, **testing**, **queue + events**, and **cron monitoring** in this Lightning-Hydra project. Assumes integrated code and configs exist; for cloning and dry-run integration, use [external-repo-integration](../external-repo-integration/SKILL.md).

## Non-negotiable rules

1. Run only one full-strength experiment on one GPU/MIG at a time.
2. Monitoring via cron + `monitor_jobs.py`, not a long-lived foreground agent loop.
3. GPU selection: prefer empty GPU; else GPU with largest free memory.
4. Full-strength runs: **`nohup`** to a log file with PID recorded for polling is the default for queue/monitor integration. If the user asks for a **detached tmux** session (attach to watch, Ctrl+B then D to leave without killing the job), follow [run_background_task](../run_background_task/SKILL.md) instead; still pass `CUDA_VISIBLE_DEVICES` and the same Hydra overrides, and keep logging if the queue expects a log path.
5. Monitoring cadence is user-defined (e.g. every 5 minutes) and implemented with cron.
6. **Crontab installation must be automated** — use [install_trial_monitor_cron.sh](../../../scripts/install_trial_monitor_cron.sh) with the plan’s `cron_monitor.sh` and interval. Do **not** instruct users to paste cron lines by hand unless the installer cannot run (then document the fallback).
7. **Benchmark / trial result tables:** prefer a **single CSV** (e.g. `benchmark_results.csv`) written by a small script or pipeline step. Do **not** require HTML or a web UI for routine reporting.
8. Full-strength `src/train.py` must include: `logger=wandb`, `+trainer.val_check_interval=<fraction>`, `++model_averaging=True`. Tune fraction by training corpus size (protocol line count): **`0.25`** mid-reference (~200k samples); **larger** corpus → **lower** fraction (e.g. ~500k → `0.125`); **smaller** corpus → **higher** fraction (e.g. ~100k → `0.5`). Interpolate between anchors.
9. **Crontab cleanup when the trial is done:** After the experiment has **finished**, **results are available** (e.g. CSV/metrics), and the **human has signed off** in `plans/<timestamp>/PLAN.md` (or equivalent checklist), **remove** the trial monitor cron entry with [remove_trial_monitor_cron.sh](../../../scripts/remove_trial_monitor_cron.sh). Do not leave `monitor_jobs` polling indefinitely after the study is closed.

## Phases

1. **Full-strength**: full data, fixed epoch budget (early stopping optional). Required Hydra flags:
   - `logger=wandb`
   - `+trainer.val_check_interval=<fraction>` (see rule 8)
   - `++model_averaging=True` (see `src/train.py`)
   - Launch (default): `nohup env CUDA_VISIBLE_DEVICES=<MIG_UUID> <run_command> > <log_file> 2>&1 &`
   - Launch (tmux, when requested): `tmux new-session -d -s <session_name> "env CUDA_VISIBLE_DEVICES=<MIG_UUID> <run_command>"` per [run_background_task](../run_background_task/SKILL.md); reply with session name, `tmux attach -t <session_name>`, and detach instructions.
   - Queue metadata: `pid`, `gpu_id` (prefer MIG UUID), `command`, `updated_at`.
2. **Testing**: evaluate against target metrics; pass/fail.
3. **Finalize**: metrics, report, memory DB update (see scripts below).

## Queue and monitor

- Human enqueues tasks; run when one GPU/MIG slot is free for a single job.
- Cron invokes `monitor_jobs.py` at the chosen interval.
- Process exits → emit `RESULT_READY` or `FAILED`.
- Heartbeat: `RUNNING` → no agent wake; `FAILED` or `NEED_DECISION` → optional agent invoke (see below; Cursor Agent uses `--print` for script mode — **do not** use `-p` as “prompt”, it is `--print`).

## Agent wake on events (two-stage)

When a trial fails or needs a decision, `monitor_jobs.py` exits **10** and the plan’s `cron_monitor.sh` (if configured) invokes `agent --continue` with `AGENT_PROMPT_FIX` (default prompt references this skill). There are **two** ways to reach that behavior:

1. **Wait for cron (reactive)**  
   - Crontab runs `plans/<timestamp>/cron_monitor.sh` every N minutes (see [install_trial_monitor_cron.sh](../../../scripts/install_trial_monitor_cron.sh)).  
   - On the **next** tick after a failure, the wrapper runs `monitor_jobs.py`; if it exits **10**, the same script starts the agent.  
   - **Latency:** up to one full cron interval (e.g. 5 minutes) before the agent runs.

2. **Proactive trigger (no wait)**  
   - From repo root, run the **same** wrapper by hand so the monitor + optional agent run **immediately**:  
     `bash plans/<timestamp>/cron_monitor.sh`  
     or equivalently: [trigger_trial_monitor_once.sh](../../../scripts/trigger_trial_monitor_once.sh) `plans/<timestamp>/cron_monitor.sh`  
   - This duplicates one cron tick: updates queue/events, and **only if** `monitor_jobs.py` exits **10** (FAILED or NEED_DECISION **in that run**), the wrapper invokes Cursor Agent. If the queue is idle/healthy, exit is **0** and **no agent runs** — that is expected.  
   - Use when you already see an error in logs and do not want to wait for the next scheduled run.  
   - Override binary or prompt if needed: `AGENT_CMD=/path/to/agent AGENT_PROMPT_FIX="/trial-experiment-monitoring …" bash plans/<timestamp>/cron_monitor.sh`  
   - **Cursor Agent CLI:** prompt is a **positional** argument after `--continue`; use `--print` for non-interactive. Example: `agent --continue --print -- "/trial-experiment-monitoring fix error and continue tasks"` (do **not** use `-p` for the prompt — `-p` is `--print`).

Do **not** spawn a second long-lived monitor process; one-shot `cron_monitor.sh` or `monitor_jobs.py` is enough.

**Verify the agent line (optional):** `cron_monitor.sh` honors `QUEUE_FILE` / `EVENTS_FILE` env overrides. With a tiny queue containing one `pending` row **without** `gpu_id` and `--gpu-id` set, `monitor_jobs.py` emits NEED_DECISION and exits **10**, so the wrapper reaches the agent command. Use `AGENT_CMD` pointing at a stub script to avoid a real agent run during checks.

### Two-stage workflow after training completion

In addition to FAILED/NEED_DECISION handling above, implement (in `cron_monitor.sh`) a two-stage event handoff driven by `events.jsonl`:

1. **Stage A (post-train eval):** when `cron_monitor.sh` observes an unprocessed `EVENT_RESULT_READY` for a given `trial_id`, it triggers the agent with a prompt like **“evaluate model for trial_id=<trial_id>”**.
   - The evaluation step should run the repo’s 5% (or target) benchmark evaluation and write evaluation artifacts to the plan output directories.
   - When evaluation finishes, the agent must append an event line to `events.jsonl` with `event` set to **`EVALUATE_MODEL_DONE`** and include at least `trial_id` and a timestamp.
2. **Stage B (summary CSV):** when `cron_monitor.sh` observes an unprocessed `EVALUATE_MODEL_DONE` event for a `trial_id`, it triggers the agent with a prompt like **“create summary report CSV for trial_id=<trial_id>”**.
   - The summary step should build a single CSV from the evaluation outputs (e.g. running `build_benchmark_results_csv.py` for that plan).
   - When the CSV is created, the agent appends `event=SUMMARY_CSV_DONE` to `events.jsonl` with `trial_id` and a timestamp.

   #### Quality gate: `eer` must not be `nan`

   - After writing the CSV, the agent must check every row’s `eer` field is a real number (not empty and not the string `nan`).
   - If any `eer` is `nan`, treat it as a **validation failure** even if the CSV exists:
     - Inspect at least one failing `(score_file, protocol_subset)` pair by re-running `scripts/score_file_to_eer.py <score> <protocol_subset>` directly and confirm:
       - protocol subset contains both `spoof` and `bonafide` labels
       - the score file parses correctly (enough numeric score fields)
       - the protocol/score merge is not empty (otherwise EER computation can yield `nan`)
     - Re-run benchmark scoring using the repo’s benchmark driver `scripts/benchmark_py/benchmark.py` (this is the “normal” path, see README usage) with the correct checkpoint path:
       - If the pretrained checkpoint cannot load, fix the checkpoint path/model config first, then re-run evaluation.
       - Use the README’s `benchmark.py` example interface (key args: `-g`, `-c`, `-b`, `-m`, `-r`, `-n`, optionally `-a` / `-l`).
     - Re-generate the summary CSV and re-check `eer` values.
   - Only append `event=SUMMARY_CSV_DONE` once the `eer` NaN check passes.

To keep the queue flowing (“next cron job will monitor next job and clean old job”):
- `cron_monitor.sh` should only trigger the agent as a short-lived subprocess for each stage (no infinite loops).
- If needed, `cron_monitor.sh` should spawn the agent asynchronously (background) so the next cron tick can continue monitoring queued trials while the agent finishes.

## Event trigger policy

- FAILED/NEED_DECISION: wake agent immediately via the existing `monitor_jobs.py` exit **10** mechanism.
- RESULT_READY: wake agent for **Stage A** (evaluate model).
- EVALUATE_MODEL_DONE: wake agent for **Stage B** (create summary report CSV).

## Runtime policy

- Default batch: `batch_size=96` (MIG-A); on OOM reduce **only** that job by `16`.
- Export `CUDA_VISIBLE_DEVICES` from `nvidia-smi` for the chosen single device before launch.
- Full-strength `<run_command>` must include the three Hydra overrides in rule 8.

## Required scripts (project)

Maintain under repository `scripts/`:

- [submit_trial.sh](../../../scripts/submit_trial.sh): create trial, enqueue
- [run_trial.py](../../../scripts/run_trial.py): dry/small test, full train, eval, finalize
- [monitor_jobs.py](../../../scripts/monitor_jobs.py): schedule, detect failure, emit events
- [finalize_trial.py](../../../scripts/finalize_trial.py): metrics, report, memory
- **[install_trial_monitor_cron.sh](../../../scripts/install_trial_monitor_cron.sh): idempotent crontab install for a `cron_monitor.sh` wrapper** (preferred over manual `crontab -e`).
- **[remove_trial_monitor_cron.sh](../../../scripts/remove_trial_monitor_cron.sh): remove that install’s cron block** when the experiment is complete and the human has approved the plan closure (see rule 9).
- **[trigger_trial_monitor_once.sh](../../../scripts/trigger_trial_monitor_once.sh): one-shot proactive run** (same as one cron tick; use when you do not want to wait for cron).

Stubs/reference: [repo-integration-trial-orchestrator/scripts/](../repo-integration-trial-orchestrator/scripts/).

## Cron setup (automated)

From repo root:

```bash
./scripts/install_trial_monitor_cron.sh plans/<timestamp>/cron_monitor.sh 5
```

- Second argument = interval in minutes (default `5` if omitted).
- Re-running replaces the previous block marked `# TRIAL_MONITOR_CRON_BEGIN` … `END`.
- Per tick: `cron_monitor.sh` runs `monitor_jobs.py`; exit **10** if `FAILED` or `NEED_DECISION` (wrapper may invoke agent continue); **0** if healthy or running.
- Robustness requirement: `cron_monitor.sh` must run any stage-trigger/event-parsing code using the repo venv interpreter (e.g. `./.venv/bin/python`), because cron PATH may not include `python`.
  - If the stage-trigger logic crashes (missing interpreter, JSON parse error, state-file write error), the wrapper must wake the agent (with a cron-bugfix prompt like `AGENT_PROMPT_CRON_BUG`) so the issue can be fixed without waiting for another unrelated failure.

Optional pattern when both monitor and launch are requested: `<cron_monitor> && <full-strength-script>` as a single one-off shell line (not cron).

## Crontab cleanup (after human sign-off)

When **all** of the following are true:

- Training/testing for this trial is **finished** (no further unattended jobs expected).
- **Results** are produced and checked (e.g. benchmark CSV, metrics, report).
- The **human** has marked the plan **done** or equivalent in `plans/<timestamp>/PLAN.md`.

Then run from repo root:

```bash
./scripts/remove_trial_monitor_cron.sh
```

This deletes only the block between `# TRIAL_MONITOR_CRON_BEGIN` and `# TRIAL_MONITOR_CRON_END` (other crontab lines are preserved). Re-install later with `install_trial_monitor_cron.sh` if a new trial needs monitoring.

## Results output (CSV)

- After score files exist, aggregate EER-related fields with `scripts/score_file_to_eer.py` per (score, protocol) pair and **write one CSV** (columns such as `benchmark_id`, `eer`, `accuracy`, `threshold`, …).
- Example plan helper: `plans/01042026/build_benchmark_results_csv.py` (adapt paths per plan).

## Output contract

```markdown
Phase: <full|test|queue|finalize>
Status: <done|running|blocked|failed|need-human-review>
Key result: <one sentence>
Next action: <one sentence>
Artifacts: <paths>
```

## Supporting files

- Detached tmux for long runs (session name, attach, Ctrl+B D): [run_background_task](../run_background_task/SKILL.md)
- Queue schema: [queue-schema.md](../repo-integration-trial-orchestrator/queue-schema.md)
- Memory template: [memory-db-template.md](../repo-integration-trial-orchestrator/memory-db-template.md)
- Cron template: [cron-template.sh](../repo-integration-trial-orchestrator/cron-template.sh)
