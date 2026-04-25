#!/usr/bin/env bash
# Remove the trial monitor cron block installed by install_trial_monitor_cron.sh (markers BEGIN/END).
# Run when the experiment is finished, results are ready, and the human has signed off in the plan.
#
# Usage:
#   ./scripts/remove_trial_monitor_cron.sh
#
set -euo pipefail

BEGIN="# TRIAL_MONITOR_CRON_BEGIN"
END="# TRIAL_MONITOR_CRON_END"

strip_block() {
  awk -v b="$BEGIN" -v e="$END" '
    $0 == b { skip=1; next }
    $0 == e { skip=0; next }
    !skip { print }
  '
}

(crontab -l 2>/dev/null || true) | strip_block | crontab -
echo "[remove_trial_monitor_cron] removed block between $BEGIN and $END (if it was present)."
if crontab -l 2>/dev/null; then
  echo "[remove_trial_monitor_cron] remaining crontab above."
else
  echo "[remove_trial_monitor_cron] crontab is now empty."
fi
