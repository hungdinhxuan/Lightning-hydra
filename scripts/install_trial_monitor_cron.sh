#!/usr/bin/env bash
# Install or refresh the trial monitor cron entry (idempotent). No manual crontab editing.
#
# Usage:
#   ./scripts/install_trial_monitor_cron.sh <path-to-cron_monitor.sh> [interval_minutes]
#
# Example:
#   ./scripts/install_trial_monitor_cron.sh plans/01042026/cron_monitor.sh 5
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER_IN="${1:?Usage: $0 <cron_monitor.sh> [interval_minutes]}"
INTERVAL="${2:-5}"

if [[ ! -f "$REPO/$WRAPPER_IN" ]] && [[ ! -f "$WRAPPER_IN" ]]; then
  echo "error: cron wrapper not found: $WRAPPER_IN" >&2
  exit 1
fi

if [[ -f "$WRAPPER_IN" ]]; then
  WRAPPER="$(realpath "$WRAPPER_IN")"
else
  WRAPPER="$(realpath "$REPO/$WRAPPER_IN")"
fi

LOG="${WRAPPER%.sh}.log"
BEGIN="# TRIAL_MONITOR_CRON_BEGIN"
END="# TRIAL_MONITOR_CRON_END"

strip_block() {
  awk -v b="$BEGIN" -v e="$END" '
    $0 == b { skip=1; next }
    $0 == e { skip=0; next }
    !skip { print }
  '
}

NEW_BLOCK=$(cat <<EOF
$BEGIN
# Lightning-hydra: monitor_jobs.py (trial-experiment-monitoring)
*/${INTERVAL} * * * * "$WRAPPER" >> "$LOG" 2>&1
$END
EOF
)

TMP="$(mktemp)"
(crontab -l 2>/dev/null | strip_block; echo "$NEW_BLOCK") | crontab -
echo "Installed cron: every ${INTERVAL} min → $WRAPPER (log: $LOG)"
crontab -l | sed -n "/$BEGIN/,/$END/p"
