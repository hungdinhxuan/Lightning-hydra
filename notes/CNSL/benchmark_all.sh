#!/usr/bin/env bash
set -euo pipefail

# Run all benchmarks in parallel with different GPU targets.
# Supports mixed MIG + non-MIG:
# - MIG: use UUID from `nvidia-smi -L` (example: MIG-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
# - Non-MIG: use numeric index (example: 0,1,2)
#
# Example:
#   bash notes/CNSL/benchmark_all.sh
#
# Optional overrides (normally not needed):
#   GPU_TARGETS="MIG-aaa...,1,MIG-bbb...,2" MAX_ACTIVE_JOBS=2 bash notes/CNSL/benchmark_all.sh

DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-512}"
BENCH_SCRIPT="./scripts/benchmark_py/benchmark.py"
RESULT_DIR="logs/results/April_2026_benchmark"
BENCH_DATA="/dev/shm/April_2026_benchmark"
MAX_RETRIES="${MAX_RETRIES:-2}"
MAX_ACTIVE_JOBS="${MAX_ACTIVE_JOBS:-0}"   # 0 means auto (use all available targets)
CHECK_IDLE_GPU="${CHECK_IDLE_GPU:-true}"  # true/false
IDLE_GPU_MAX_UTIL="${IDLE_GPU_MAX_UTIL:-10}"      # percent
IDLE_GPU_MAX_MEM_MB="${IDLE_GPU_MAX_MEM_MB:-2048}" # MiB
FALLBACK_ACTIVE_JOBS="${FALLBACK_ACTIVE_JOBS:-1}"  # when no idle target found
LOCK_FILE="${LOCK_FILE:-/tmp/lightning_hydra_benchmark_all.lock}"

# Prevent overlapping runs of this orchestrator script.
exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
  echo "ERROR: benchmark_all.sh is already running (lock: ${LOCK_FILE})."
  echo "Stop the existing run first, then re-run this script."
  exit 1
fi

declare -a GPU_TARGETS_ARR=()

to_lower() {
  echo "$1" | tr '[:upper:]' '[:lower:]'
}

is_numeric_gpu_target() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

is_gpu_idle() {
  local gpu_target="$1"
  local line mem util proc_count

  # Primary signal: no compute process attached to this target.
  # This is reliable for both MIG UUID and non-MIG index.
  if is_numeric_gpu_target "${gpu_target}"; then
    proc_count="$(nvidia-smi -i "${gpu_target}" \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk '/^[0-9]+$/ {count++} END {print count+0}')"
  else
    # MIG UUID does not support -i directly in current nvidia-smi.
    # Keep process count as unknown here; rely on telemetry checks below.
    proc_count="-1"
  fi
  if [[ "${proc_count}" =~ ^[0-9]+$ ]] && (( proc_count == 0 )); then
    return 0
  fi

  # Works for both MIG UUID and non-MIG index because CUDA_VISIBLE_DEVICES
  # isolates the target as visible device 0 for this command.
  if is_numeric_gpu_target "${gpu_target}"; then
    line="$(nvidia-smi -i "${gpu_target}" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {print $0}')"
  else
    line="$(CUDA_VISIBLE_DEVICES="${gpu_target}" nvidia-smi \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {print $0}')"
  fi

  if [[ -z "${line}" ]]; then
    return 1
  fi

  mem="$(echo "${line}" | awk -F',' '{gsub(/ /,"",$1); print $1}')"
  util="$(echo "${line}" | awk -F',' '{gsub(/ /,"",$2); print $2}')"

  # Some MIG devices may report [N/A] for utilization/memory.
  if [[ ! "${mem}" =~ ^[0-9]+$ ]]; then
    return 1
  fi

  if [[ -z "${mem}" ]]; then
    return 1
  fi

  # For MIG targets utilization can be N/A; memory is the primary signal there.
  if [[ ! "${util}" =~ ^[0-9]+$ ]]; then
    if (( mem <= IDLE_GPU_MAX_MEM_MB )); then
      return 0
    fi
    return 1
  fi

  if (( mem <= IDLE_GPU_MAX_MEM_MB && util <= IDLE_GPU_MAX_UTIL )); then
    return 0
  fi
  return 1
}

get_gpu_mem_used() {
  local gpu_target="$1"
  local mem_line mem
  if is_numeric_gpu_target "${gpu_target}"; then
    mem_line="$(nvidia-smi -i "${gpu_target}" \
      --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {print $0}')"
  else
    mem_line="$(CUDA_VISIBLE_DEVICES="${gpu_target}" nvidia-smi \
      --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | awk 'NR==1 {print $0}')"
  fi
  mem="$(echo "${mem_line}" | awk '{gsub(/ /,"",$1); print $1}')"
  if [[ -z "${mem}" || ! "${mem}" =~ ^[0-9]+$ ]]; then
    echo "999999"
  else
    echo "${mem}"
  fi
}

get_gpu_proc_count() {
  local gpu_target="$1"
  local proc_count
  if is_numeric_gpu_target "${gpu_target}"; then
    proc_count="$(nvidia-smi -i "${gpu_target}" \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null \
      | awk '/^[0-9]+$/ {count++} END {print count+0}')"
  else
    # MIG UUID process count cannot be queried directly by nvidia-smi -i.
    proc_count="999999"
  fi
  if [[ "${proc_count}" =~ ^[0-9]+$ ]]; then
    echo "${proc_count}"
  else
    echo "999999"
  fi
}

select_usable_gpu_targets() {
  local do_check
  do_check="$(to_lower "${CHECK_IDLE_GPU}")"
  if [[ "${do_check}" != "true" ]]; then
    USABLE_GPU_TARGETS=("${GPU_TARGETS_ARR[@]}")
    return
  fi

  USABLE_GPU_TARGETS=()
  for target in "${GPU_TARGETS_ARR[@]}"; do
    if is_gpu_idle "${target}"; then
      USABLE_GPU_TARGETS+=("${target}")
    fi
  done

  # Fallback: if no target passes idle check, keep original list so script can still run.
  if [[ "${#USABLE_GPU_TARGETS[@]}" -eq 0 ]]; then
    local -a ranked=()
    local target mem proc_count fallback_count
    fallback_count="${FALLBACK_ACTIVE_JOBS}"
    if [[ "${fallback_count}" -lt 1 ]]; then
      fallback_count=1
    fi
    if [[ "${fallback_count}" -gt "${#GPU_TARGETS_ARR[@]}" ]]; then
      fallback_count="${#GPU_TARGETS_ARR[@]}"
    fi

    for target in "${GPU_TARGETS_ARR[@]}"; do
      proc_count="$(get_gpu_proc_count "${target}")"
      mem="$(get_gpu_mem_used "${target}")"
      ranked+=("${proc_count}|${mem}|${target}")
    done

    # Sort by fewest compute processes first, then lower memory used.
    mapfile -t ranked < <(printf "%s\n" "${ranked[@]}" | sort -n -t'|' -k1,1 -k2,2)

    USABLE_GPU_TARGETS=()
    local i chosen
    for ((i = 0; i < fallback_count; i++)); do
      chosen="$(echo "${ranked[$i]}" | awk -F'|' '{print $3}')"
      [[ -n "${chosen}" ]] && USABLE_GPU_TARGETS+=("${chosen}")
    done

    echo "[WARN ] No idle GPU target found."
    echo "[WARN ] Fallback to least-busy target(s): ${USABLE_GPU_TARGETS[*]}"
  fi
}

discover_gpu_targets() {
  local csv_targets="${GPU_TARGETS:-}"
  local -a discovered_mig=()
  local -a discovered_non_mig=()
  local line

  # Respect explicit override if user still wants manual control.
  if [[ -n "${csv_targets}" ]]; then
    IFS=',' read -r -a GPU_TARGETS_ARR <<< "${csv_targets}"
    return
  fi

  # 1) Collect MIG device UUIDs.
  while IFS= read -r line; do
    [[ -n "${line}" ]] && discovered_mig+=("${line}")
  done < <(
    nvidia-smi -L 2>/dev/null \
      | awk -F'UUID: ' '/MIG/ {gsub(/\)/, "", $2); gsub(/ /, "", $2); print $2}'
  )

  # 2) Collect non-MIG GPU indices only.
  while IFS= read -r line; do
    [[ -n "${line}" ]] && discovered_non_mig+=("${line}")
  done < <(
    nvidia-smi --query-gpu=index,mig.mode.current --format=csv,noheader 2>/dev/null \
      | awk -F',' '
        {
          idx=$1; mode=$2;
          gsub(/ /, "", idx);
          gsub(/ /, "", mode);
          if (tolower(mode) != "enabled") print idx;
        }'
  )

  GPU_TARGETS_ARR=("${discovered_mig[@]}" "${discovered_non_mig[@]}")

  if [[ "${#GPU_TARGETS_ARR[@]}" -lt 1 ]]; then
    echo "ERROR: Could not auto-discover any GPU target."
    echo "Hint: pass manual targets, e.g. GPU_TARGETS=\"0,1\" bash notes/CNSL/benchmark_all.sh"
    exit 1
  fi
}

run_job() {
  local job_name="$1"
  local gpu_target="$2"
  local start_batch="$3"
  shift 3

  local try_idx=1
  local batch_size="${start_batch}"
  while true; do
    echo "[START] ${job_name} on GPU target: ${gpu_target} (batch=${batch_size}, try=${try_idx})"
    # Pass target directly to benchmark script:
    # - MIG idle target: -g <MIG_UUID>
    # - non-MIG target:  -g <physical GPU index>
    if DEFAULT_BATCH_SIZE="${batch_size}" uv run "${BENCH_SCRIPT}" -g "${gpu_target}" "$@"; then
      echo "[DONE ] ${job_name} (batch=${batch_size}, try=${try_idx})"
      return 0
    fi

    if [[ "${try_idx}" -gt "${MAX_RETRIES}" ]]; then
      echo "[FAIL ] ${job_name} after $((MAX_RETRIES + 1)) attempts"
      return 1
    fi

    batch_size=$((batch_size / 2))
    if [[ "${batch_size}" -lt 32 ]]; then
      batch_size=32
    fi
    try_idx=$((try_idx + 1))
    echo "[RETRY] ${job_name} with reduced batch=${batch_size}"
    sleep 2
  done
}

discover_gpu_targets
select_usable_gpu_targets
if [[ "${MAX_ACTIVE_JOBS}" -le 0 ]]; then
  MAX_ACTIVE_JOBS="${#USABLE_GPU_TARGETS[@]}"
fi
if [[ "${MAX_ACTIVE_JOBS}" -lt 1 ]]; then
  MAX_ACTIVE_JOBS=1
fi
if [[ "${MAX_ACTIVE_JOBS}" -gt "${#USABLE_GPU_TARGETS[@]}" ]]; then
  MAX_ACTIVE_JOBS="${#USABLE_GPU_TARGETS[@]}"
fi

echo "[INFO ] Provided GPU targets : ${GPU_TARGETS_ARR[*]}"
echo "[INFO ] Usable GPU targets   : ${USABLE_GPU_TARGETS[*]}"
echo "[INFO ] MAX_ACTIVE_JOBS      : ${MAX_ACTIVE_JOBS}"

declare -a pids=()
active_jobs=0
next_gpu_idx=0
exit_code=0

launch_job() {
  local job_name="$1"
  local start_batch="$2"
  shift 2

  local gpu_slot
  gpu_slot=$((next_gpu_idx % MAX_ACTIVE_JOBS))
  local gpu_target="${USABLE_GPU_TARGETS[${gpu_slot}]}"

  run_job "${job_name}" "${gpu_target}" "${start_batch}" "$@" &
  pids+=($!)
  active_jobs=$((active_jobs + 1))
  next_gpu_idx=$((next_gpu_idx + 1))

  if (( active_jobs >= MAX_ACTIVE_JOBS )); then
    if ! wait -n; then
      exit_code=1
    fi
    active_jobs=$((active_jobs - 1))
  fi
}

# S_241214_conf-1
launch_job "S_241214_conf-1" "${BATCH_S241:-${DEFAULT_BATCH_SIZE}}" \
  -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer \
  -b "${BENCH_DATA}" \
  -m pretrained/S_241214_conf-1.pth \
  -r "${RESULT_DIR}" \
  -n "S_241214_conf-1" \
  -l false

# LoRAMay2025
launch_job "LoRAMay2025" "${BATCH_LORA_MAY2025:-${DEFAULT_BATCH_SIZE}}" \
  -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_more_elevenlabs \
  -b "${BENCH_DATA}" \
  -m pretrained/S_241214_conf-1.pth \
  -a pretrained/MDT_241214_lora_250501 \
  -r "${RESULT_DIR}" \
  -n "LoRAMay2025" \
  -l false

# Feb 2026
launch_job "06feb26_xlsr_conformertcm_mdt_vad" "${BATCH_FEB2026:-${DEFAULT_BATCH_SIZE}}" \
  -c cnsl/lora/elevenlabs/xlsr_conformertcm_mdt_lora_infer \
  -b "${BENCH_DATA}" \
  -m /NAS1_pretrained_lab/06feb26_xlsr_conformertcm_mdt_vad.pt \
  -r "${RESULT_DIR}" \
  -n "06feb26_xlsr_conformertcm_mdt_vad" \
  -l false

# VIB-April
launch_job "VIB_April" "${BATCH_VIB_APRIL:-256}" \
  -c cnsl/xlsr_vib_large_corpus \
  -b "${BENCH_DATA}" \
  -m /home/hungdx/code/Lightning-hydra/pretrained/vib_conf-5_gelu_acmccs_apr3_moreko_telephone_epoch22.pth \
  -r "${RESULT_DIR}" \
  -n "VIB_April" \
  -l false

while (( active_jobs > 0 )); do
  if ! wait -n; then
    exit_code=1
  fi
  active_jobs=$((active_jobs - 1))
done

exit "${exit_code}"