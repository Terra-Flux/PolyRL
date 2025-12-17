#!/usr/bin/env bash

set -euo pipefail

# Launch one SGLang server every LAUNCH_INTERVAL seconds. Once MAX_INSTANCES are
# up, stop the oldest instance before starting a new one so the fleet rolls
# forward continually.
# NOTE: memory fraction is set to 0.6; 0.9 causes OOM due to metadata volume.

readonly ROLLOUT_MANAGER_ADDR="http://$1:$2"
readonly HOST_ADDR="$(hostname -i || echo 127.0.0.1)"
readonly MODEL_PATH="Qwen/Qwen3-8B"

readonly GPU_SETS=("0,1" "2,3" "4,5" "6,7")
readonly PORTS=(40001 40002 40003 40004)
readonly HANDSHAKE_PORTS=(20000 21000 22000 23000)
readonly MAX_INSTANCES="${#GPU_SETS[@]}"
readonly LAUNCH_INTERVAL=1800 # seconds
readonly LOG_DIR="${LOG_DIR:-./logs/sglang}"

declare -a PIDS=()
trap 'echo "Stopping all instances..."; kill "${PIDS[@]}" 2>/dev/null || true' EXIT

log() {
  echo "[$(date --iso-8601=seconds)] $*"
}

start_instance() {
  local slot="$1"
  local gpus="${GPU_SETS[$slot]}"
  local port="${PORTS[$slot]}"
  local handshake="${HANDSHAKE_PORTS[$slot]}"
  local ts
  ts="$(date +%Y%m%d-%H%M%S)"
  local logfile="${LOG_DIR}/sglang_slot${slot}_port${port}_${ts}.log"

  mkdir -p "${LOG_DIR}"

  log "Starting SGLang slot=${slot} GPUs=${gpus} port=${port} handshake=${handshake} -> ${logfile}"
  CUDA_VISIBLE_DEVICES="${gpus}" python -m rlboost.sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host "${HOST_ADDR}" \
    --port "${port}" \
    --grammar-backend outlines \
    --tp-size 2 \
    --mem-fraction-static 0.6 \
    --max-running-requests 256 \
    --stream-interval 10 \
    --enable-mixed-chunk \
    --enable-weight-transfer-agent \
    --transfer-agent-handshake-port "${handshake}" \
    --rollout-manager-address "${ROLLOUT_MANAGER_ADDR}" \
    >"${logfile}" 2>&1 &

  local pid=$!
  PIDS+=("${pid}")
  log "Started pid=${pid} (instances running: ${#PIDS[@]}/${MAX_INSTANCES})"
}

stop_oldest() {
  local pid="${PIDS[0]}"
  log "Stopping oldest instance pid=${pid}"
  kill -9 "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  PIDS=("${PIDS[@]:1}")
}

main() {
  local slot=0
  while true; do
    if ((${#PIDS[@]} >= MAX_INSTANCES)); then
      stop_oldest
    fi

    start_instance $((slot % MAX_INSTANCES))
    slot=$((slot + 1))

    sleep "${LAUNCH_INTERVAL}"
  done
}

main
