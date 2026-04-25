---
name: run_headless_agent
description: Runs the Cursor CLI agent headlessly in a detached tmux session with auto-approved edits (-p --force) so unattended runs never block on prompts. Does not use for instant commands (ls, cat, pwd). Use when the user asks to run a long agent task in the background, detach an agent session, or keep an automated agent loop alive without interactive approval.
---

# Run headless agent (tmux + Cursor CLI)

## When to apply

- **Use** for long-running agent work: multi-step refactors, experiments driven by the agent, unattended automation.
- **Do not use** for trivial or instant commands. Run those in the foreground.

For long **non-agent** shell commands in tmux only, use [run_background_task](../run_background_task/SKILL.md).

## Workflow

1. **Session name**: Short, unique, no spaces (e.g. `agent-exp_01`, `fix-bench-042`).
2. **Execute** in a detached tmux session. Set `AGENT_BIN` to the Cursor agent executable (e.g. `agent` or an absolute path). **Must** pass **`-p`** and **`--force`** so the agent auto-approves file changes and does not block on user input.

   Exact pattern:

   ```bash
   tmux new-session -d -s <session_name> "$AGENT_BIN -p --force \"<the_user_prompt_or_command>\""
   ```

   If `<the_user_prompt_or_command>` contains double quotes, escape them or pass the inner command via a wrapper script so the shell string stays valid.

## Reply to the user (exactly three items)

1. The **session name** created.
2. **Monitor**: `tmux attach -t <session_name>`
3. **Detach without stopping**: To exit without stopping the task, press Ctrl+B, then D.

Optional troubleshooting only if asked: `tmux ls`, `tmux kill-session -t <session_name>`.
