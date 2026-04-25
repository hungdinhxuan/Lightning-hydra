#!/usr/bin/env bash
# One-shot trial monitor: same as a single cron tick (monitor_jobs.py, then agent if exit 10).
# Use for proactive recovery without waiting for the next crontab run.
#
# Usage:
#   ./scripts/trigger_trial_monitor_once.sh plans/<timestamp>/cron_monitor.sh
#
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER_IN="${1:?usage: $0 plans/<timestamp>/cron_monitor.sh}"

if [[ -f "$REPO/$WRAPPER_IN" ]]; then
  exec bash "$REPO/$WRAPPER_IN"
elif [[ -f "$WRAPPER_IN" ]]; then
  exec bash "$WRAPPER_IN"
else
  echo "error: cron wrapper not found: $WRAPPER_IN" >&2
  exit 1
fi
