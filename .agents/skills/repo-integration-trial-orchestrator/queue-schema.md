# Queue Schema

Minimal JSONL queue record:

```json
{
  "trial_id": "trial-20260401-001",
  "owner": "human_name",
  "experiment": "external_repo_integration_a",
  "status": "pending",
  "priority": 5,
  "gpu_id": null,
  "batch_size": 96,
  "command": "python scripts/run_trial.py --trial-id trial-20260401-001",
  "target_metric": {"name": "eer", "threshold": 0.08},
  "updated_at": "2026-04-01T12:00:00Z"
}
```

Status values:
- `pending`
- `running`
- `blocked`
- `failed`
- `done`
