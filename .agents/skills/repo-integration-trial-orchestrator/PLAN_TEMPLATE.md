# PLAN Template

Use this file as `plans/<DDMMYYYY>/PLAN.md`.

```markdown
# Integration Plan - <DDMMYYYY>

## Scope
- External repo: <url>
- Reference path: references/<new_repo_name>
- Target integration area(s): <paths/modules>

## Source Mapping
- Model definition: <file>
- Config system: <file(s)>
- Dataset + dataloader: <file(s)>
- Augmentation: <file(s)>
- Training method (optional): <file(s)>

## Planned Changes
- [ ] Edit <file>: <reason>
- [ ] Edit <file>: <reason>
- [ ] Add <file>: <reason>

## Data Contract Validation
- [ ] Training dataset path verified
- [ ] Training protocol path verified
- [ ] Benchmark root verified
- [ ] Benchmark protocol files verified
- [ ] Protocol format validated: <file_path> <subset> <label>

## Experiment Plan
- Dry-run: <subset/epochs/objective>
- Full-strength: <epochs/early-stop policy>
- Testing: <metrics + thresholds>

## Runtime Policy
- Initial batch size (MIG-A): 96
- OOM fallback for failing job only: -16 per retry

## Required Script Plan
- [ ] submit_trial.sh
- [ ] run_trial.py
- [ ] monitor_jobs.py
- [ ] finalize_trial.py

## Risks / Open Questions
- <question 1>
- <question 2>

## Human Review
- Status: PENDING
- Human comments:
  - <to be filled by reviewer>
```
