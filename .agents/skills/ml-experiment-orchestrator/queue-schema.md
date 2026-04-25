# Queue Schema

Minimal JSONL queue record:

```json
{
  "task_id": "exp-2026-03-31-001",
  "owner": "human_name",
  "experiment": "model_variant_a",
  "status": "pending",
  "priority": 5,
  "gpu_id": null,
  "command": "python train.py --config configs/exp_a.yaml",
  "target_metric": {"name": "eer", "threshold": 0.08},
  "updated_at": "2026-03-31T12:00:00Z"
}
```

## Status Rules

- `pending`: waiting for GPU and trigger.
- `running`: currently executing.
- `blocked`: waiting for human input (e.g., data-prep script).
- `failed`: run ended with error.
- `done`: completed and evaluated.

## Selection Policy

1. Highest `priority` first.
2. Oldest `updated_at` first on ties.
3. Skip tasks with unmet prerequisites.
