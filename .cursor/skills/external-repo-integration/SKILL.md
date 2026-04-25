---
name: external-repo-integration
description: Clones external ML repositories into references/, checks environment compatibility against README, maps model/config/data/augmentation code, authors human-approved plans under plans/<timestamp>/, integrates code into the Lightning-Hydra tree, validates space-separated protocols, and verifies with a GPU dry-run. Use when onboarding a new training repository, porting external model code into this project, or before any full-strength training; not for queue/cron/full-strength launch (see trial-experiment-monitoring).
---

# External repo integration (code + dry-run)

## Purpose

Use for **bringing external training code into this codebase** and **proving the integrated pipeline with a dry-run**. Does **not** define full-strength launch, queue files, or cron monitoring; use [trial-experiment-monitoring](../trial-experiment-monitoring/SKILL.md) after dry-run succeeds.

## Required inputs

- `new_repo_url`: Git URL to clone.
- `new_repo_name`: Folder name under `references/`.
- `timestamp`: `DDMMYYYY` for `plans/<timestamp>/`.

Optional: target metric(s), branch/tag/commit of external repo.

## Non-negotiable rules

1. Clone external code into `references/<new_repo_name>` only.
2. Read `references/<new_repo_name>/README.md` before any install.
3. Compare README requirements vs current environment first.
4. If environment conflicts exist, stop and request manual human installation.
5. If no conflict, install into `.venv`.
6. Create `plans/<timestamp>/PLAN.md` and iterate until human approval.
7. Do not execute integration edits before plan approval.
8. Validate protocol files use `<file_path> <subset> <label>` (space-separated).
9. For dry-run, set `CUDA_VISIBLE_DEVICES` to exactly one MIG device; derive IDs from `nvidia-smi -L`, never guess.
10. Run only one dry-run at a time on one GPU/MIG unless explicitly approved.

## Workflow checklist

```markdown
Integration (code) Progress:
- [ ] 1) Clone repo to references/<new_repo_name>
- [ ] 2) Read README and check environment compatibility
- [ ] 3) Install dependencies in .venv (only if no conflict)
- [ ] 4) Scan source for model/config/data/augmentation/training logic
- [ ] 5) Create plans/<timestamp>/PLAN.md from template
- [ ] 6) Human review + revise PLAN.md until approved
- [ ] 7) Integrate into main source tree + configs
- [ ] 8) Dry-run: small data, low epochs, sanity checks
```

## Step details

### Clone

- Target: `references/<new_repo_name>`. If exists, ask reuse vs replace vs new suffix.
- Record URL and commit hash in `plans/<timestamp>/PLAN.md`.

### README and environment

- Extract Python/CUDA/deps from README; compare with project env.
- Conflict → stop for human. No conflict → `.venv`.

### Source scan

Record paths for: model, config, dataset/loader, augmentation, training entrypoints.

### Plan loop

- Author from [PLAN_TEMPLATE.md](../repo-integration-trial-orchestrator/PLAN_TEMPLATE.md).
- Include files to change, migration notes, tests. No implementation until approved.

### Integration constraints

- Follow `Project_Structure.md`.
- Data contract in `plans/<timestamp>/DATA.md`: training datadir + protocol; benchmark roots + per-dataset `protocol.txt`.
- Protocol lines: `<file_path> <subset> <label>`, spaces only.

### Dry-run

- `nvidia-smi -L` → pick one MIG UUID; `CUDA_VISIBLE_DEVICES=<that_id>`.
- Small subset, `trainer.max_epochs` / `limit_*_batches` as appropriate; `test=false` if test config is not ready.
- Goal: forward pass, dataloader, and training step sanity—not production training.

## Handoff

When dry-run passes and integration is stable, use **trial-experiment-monitoring** for full-strength commands (`logger=wandb`, `+trainer.val_check_interval`, `++model_averaging=True`), `nohup`, queue, `monitor_jobs.py`, and cron.

## Supporting files

- Plan starter: [PLAN_TEMPLATE.md](../repo-integration-trial-orchestrator/PLAN_TEMPLATE.md)
