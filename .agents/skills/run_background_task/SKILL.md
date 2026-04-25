---
name: run_background_task
description: Runs long-running experiments or scripts in a detached tmux session with a unique name. Do not use for quick commands (ls, cat, pwd, git status, etc.). Use when the user asks to run something in the background, detach a long job, or keep a training/evaluation script alive after the terminal closes.
---

# Run background task (tmux)

## When to apply

- **Use** for long-running jobs: training, evaluation, downloads, builds, batch scripts, servers.
- **Do not use** for trivial or instant commands. If it finishes in seconds, run it in the foreground instead.

In this repo, **full-strength Lightning-Hydra trials** with queue/cron usually default to `nohup` per [trial-experiment-monitoring](../trial-experiment-monitoring/SKILL.md); use this skill when the user explicitly wants **tmux** (attach/detach) for those or other long commands.

For the **Cursor CLI agent** headless with `-p` and `--force`, use [run_headless_agent](../run_headless_agent/SKILL.md).

## Workflow

1. **Session name**: Generate a short, unique name (no spaces), e.g. `agent_exp_01`, `train_wildspoof_042`.
2. **Execute** with this exact pattern (double quotes around the full command string):

   ```bash
   tmux new-session -d -s <session_name> "<command>"
   ```

   If `<command>` contains double quotes, escape them or refactor (e.g. wrap inner parts in single quotes) so the outer string passed to tmux remains valid.

## Reply to the user (exactly three items)

1. The **session name** created.
2. **Monitor**: `tmux attach -t <session_name>`
3. **Detach without stopping**: To exit without stopping the task, press Ctrl+B, then D.

No extra sections unless the user asks for troubleshooting (e.g. `tmux ls`, `tmux kill-session -t <session_name>`).
