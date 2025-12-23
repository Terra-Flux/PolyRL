#!/bin/bash

# NOTE: 
# 1. memory fraction is set to 0.6, 0.9 will cause OOM because of the large amount of meta data through requests
HOST_ADDR="$(hostname -i)"
HOST_PORT=$1
HANDSHAKE_PORT=$2
ROLLOUT_MANAGER_ADDR="http://localhost:5000"

python -m rlboost.sglang.launch_server \
    --model-path Qwen/Qwen3-1.7B \
    --host $HOST_ADDR \
    --port $HOST_PORT \
    --grammar-backend outlines \
    --tp-size 2 \
    --mem-fraction-static 0.6 \
    --max-running-requests 256 \
    --stream-interval 10 \
    --enable-mixed-chunk \
    --enable-weight-transfer-agent \
    --transfer-agent-handshake-port $HANDSHAKE_PORT \
    --rollout-manager-address "${ROLLOUT_MANAGER_ADDR}"
