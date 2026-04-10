---
name: ml-experiment-orchestrator
description: Orchestrates long-running ML experiments with human-in-the-loop data preparation, dry-run tuning, full-strength training, testing gates, GPU queue execution, cron-based agent triggering, and compact memory database updates. Use when users ask to plan, run, monitor, or resume training pipelines on shared GPU servers.
---

# ML Experiment Orchestrator

## Purpose

Use this skill for end-to-end experiment execution where:
- Humans provide or validate dataset preparation scripts.
- AI runs experiments in phases.
- Tasks are queued and scheduled on available GPUs.
- Long runs are monitored via cron triggers.
- Each major step is summarized into a compact memory database.

## Operating Rules

1. Never invent data preprocessing details. Ask for or reuse a human-provided script.
2. Always run a **dry-run phase** before full training.
3. Promote to **full-strength phase** only after dry-run metrics are logged.
4. Run **testing phase** against explicit target metrics.
5. Use a queue entry for each experiment run.
6. After each major step, append a compact memory note.

## Workflow

Copy and track this checklist:

```markdown
Experiment Progress:
- [ ] 1) Confirm task + target metric
- [ ] 2) Get or validate human data-prep script
- [ ] 3) Register queue task
- [ ] 4) Dry-run on small data / low epochs
- [ ] 5) Select hyperparameters
- [ ] 6) Full-strength training on full data
- [ ] 7) Testing and pass/fail decision
- [ ] 8) Update memory DB summary
- [ ] 9) Mark queue task done or blocked
```

### 1) Data-processing (human-in-the-loop)

- Require a human-provided script path (or a partially completed script to finish).
- Validate script interface:
  - input dataset location
  - output dataset/artifact location
  - deterministic split or seed handling
- Do not start model training until data-prep output exists.

### 2) Dry-run phase

- Use a reduced dataset slice and small epoch count.
- Objective: hyperparameter exploration and pipeline sanity checks.
- Log:
  - chosen subset rule
  - candidate hyperparameters
  - top dry-run result
  - failure reasons (if any)

### 3) Full-strength phase

- Use full dataset and fixed epoch budget.
- Early stopping is allowed if configured.
- Freeze core hyperparameters selected from dry-run.
- Save checkpoints, config snapshot, and training summary.

### 4) Testing phase

- Evaluate on held-out test split.
- Compare results against explicit target metric(s).
- Output one of:
  - `PASS`: target met
  - `FAIL`: target not met
  - `INCONCLUSIVE`: run invalid or incomplete

### 5) Queue system

- Human adds queue items.
- AI polls queue for `pending` tasks.
- AI selects an available GPU and starts the run.
- Update queue status transitions:
  - `pending -> running -> done`
  - `pending -> running -> blocked`
  - `running -> failed`

Queue fields should include:
- task id
- owner/requester
- model/experiment name
- status
- gpu id (nullable until assigned)
- priority
- command
- last update timestamp

### 6) Cron-tab strategy

Use cron to trigger lightweight monitoring and deferred AI activation.

- Frequent monitor: queue scan + GPU availability check.
- Trigger condition: pending task exists AND free GPU exists.
- Expensive AI steps should run only when trigger condition is true.

Use [cron-template.sh](cron-template.sh) as baseline.

### 7) Memory database

After every major step, write a compact note with:
- current phase
- decisions made
- artifacts produced
- blockers
- next action

Use [memory-db-template.md](memory-db-template.md) to keep context short and resumable.

## Output Contract

When reporting progress, use this format:

```markdown
Phase: <data-prep|dry-run|full-strength|testing|queue|monitoring>
Status: <done|running|blocked|failed>
Key result: <1 sentence>
Next action: <1 sentence>
Artifacts: <paths or IDs>
```

## Supporting Files

- Queue schema and example: [queue-schema.md](queue-schema.md)
- Memory note format: [memory-db-template.md](memory-db-template.md)
- Cron bootstrap script: [cron-template.sh](cron-template.sh)
