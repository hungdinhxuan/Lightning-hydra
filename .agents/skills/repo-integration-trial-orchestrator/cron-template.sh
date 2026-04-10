#!/usr/bin/env bash
set -euo pipefail

QUEUE_FILE="${QUEUE_FILE:-queue.jsonl}"
LOCK_FILE="${LOCK_FILE:-/tmp/repo-integration-trial.lock}"
MONITOR_CMD="${MONITOR_CMD:-python scripts/monitor_jobs.py --queue-file "$QUEUE_FILE"}"
EVENTS_FILE="${EVENTS_FILE:-events.jsonl}"

if [[ -f "$LOCK_FILE" ]]; then
  exit 0
fi

touch "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

has_pending="$(
  python - <<'PY'
import json
import os
from pathlib import Path
q = Path(os.environ["QUEUE_FILE"])
if not q.exists():
    print("no")
else:
    rows = [json.loads(x) for x in q.read_text().splitlines() if x.strip()]
    print("yes" if any(r.get("status") == "pending" for r in rows) else "no")
PY
)"

if [[ "$has_pending" == "yes" ]]; then
  # Select GPU policy:
  # 1) Prefer empty GPU (no running compute process)
  # 2) If all GPUs are busy, pick GPU with largest free memory
  selected_gpu="$(
    nvidia-smi --query-gpu=index,pci.bus_id,memory.free --format=csv,noheader,nounits | \
    awk -F', *' '
      FNR==NR {
        busy[$1]=1
        next
      }
      {
        idx=$1
        bus=$2
        mem=$3+0
        if (!(bus in busy)) {
          if (mem > best_empty_mem) {
            best_empty_mem=mem
            best_empty_idx=idx
          }
        }
        if (mem > best_any_mem) {
          best_any_mem=mem
          best_any_idx=idx
        }
      }
      END {
        if (best_empty_idx != "") {
          print best_empty_idx
        } else {
          print best_any_idx
        }
      }
    ' <(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader 2>/dev/null | sed "/^$/d")
  )"

  if [[ -n "${selected_gpu:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$selected_gpu"
    # monitor_jobs.py should launch full-strength with nohup (or equivalent),
    # persist PID to queue, and let cron poll process status by PID.
    eval "$MONITOR_CMD --gpu-id $selected_gpu"

    latest_event="$(python - <<'PY'
import json, os
from pathlib import Path
ev = Path(os.environ["EVENTS_FILE"])
if not ev.exists() or not ev.read_text().strip():
    print("RUNNING")
else:
    last = json.loads([x for x in ev.read_text().splitlines() if x.strip()][-1])
    print(last.get("event", "RUNNING"))
PY
)"

    if [[ "$latest_event" == "FAILED" || "$latest_event" == "NEED_DECISION" ]]; then
      agent --continue -p "/repo-integration-trial-orchestrator fix error and continue tasks"
    fi
  fi
fi
