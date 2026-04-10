---
name: repo-integration-trial-orchestrator
description: Legacy umbrella for Lightning-hydra external-repo onboarding plus trial execution. This workflow is split: use external-repo-integration for clone/plan/integration/dry-run, and trial-experiment-monitoring for full-strength runs, queue, and cron. Use when the user mentions the old combined skill name or wants an end-to-end pointer without choosing a phase.
---

# Repo integration + trial orchestrator (split)

The former monolithic workflow is now **two skills**:

| Phase | Skill |
|--------|--------|
| Clone, plan, integrate external code, protocol validation, **dry-run** | [external-repo-integration/SKILL.md](external-repo-integration/SKILL.md) |
| **Full-strength** training, testing, queue, `monitor_jobs.py`, cron, finalize | [trial-experiment-monitoring/SKILL.md](trial-experiment-monitoring/SKILL.md) |

For monitor/cron recovery prompts, prefer `@trial-experiment-monitoring` (or `-p "/trial-experiment-monitoring …"`). The old combined name still resolves to this router.

**Shared assets** (templates, queue docs, script stubs) remain in this directory:

- [PLAN_TEMPLATE.md](PLAN_TEMPLATE.md)
- [queue-schema.md](queue-schema.md)
- [memory-db-template.md](memory-db-template.md)
- [cron-template.sh](cron-template.sh)
- [scripts/](scripts/)

**Order of operations:** complete **external-repo-integration** (including dry-run) before relying on **trial-experiment-monitoring** for production runs.
