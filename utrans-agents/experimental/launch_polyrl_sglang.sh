#!/bin/bash

# NOTE: 
# 1. memory fraction is set to 0.6, 0.9 will cause OOM because of the large amount of meta data through requests
HOST_ADDR="$(hostname -i)"
ROLLOUT_MANAGER_ADDR="http://10.202.15.230:8000"

# 1. Select the specific chips
export TPU_VISIBLE_DEVICES=0,1,2,3

# 2. Define the new topology (2x2 square = 4 chips)
export TPU_CHIPS_PER_HOST_BOUNDS=1,4,1
export TPU_HOST_BOUNDS=1,1,1

# 3. (Optional) Set a unique ID/Port to avoid conflicts if running multiple jobs
export TPU_MESH_CONTROLLER_ADDRESS=localhost:8476
export TPU_MESH_CONTROLLER_PORT=8476

export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache

uv run python -u -m patch.sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --trust-remote-code  \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1  \
    --tp-size=4 \
    --device=tpu \
    --node-rank=0 \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=32768 \
    --chunked-prefill-size=8192 \
    --context-len 32768 \
    --download-dir=/tmp \
    --stream-output \
    --log-requests-level=2 \
    --dtype="auto"  \
    --host $HOST_ADDR \
    --port "$1" \
    --page-size 128 \
    --max-running-requests=256 \
    --stream-interval 32 \
    --enable-weight-transfer-agent \
    --rollout-manager-address "${ROLLOUT_MANAGER_ADDR}" \
    --transfer-agent-handshake-port "$2" \
    --disable-overlap-schedule


