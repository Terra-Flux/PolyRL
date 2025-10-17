#!/bin/bash

# NOTE: 
# 1. memory fraction is set to 0.6, 0.9 will cause OOM because of the large amount of meta data through requests
ROLLOUT_MANAGER_ADDR="<ROLLOUT_MANAGER_IP>:<PORT>"
HOST_ADDR="<YOUR_HOST_IP>"

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-1.7B \
    --port 40000 \
    --host $HOST_ADDR \
    --mooncake-handshake-port 20000 \
    --grammar-backend outlines \
    --tp-size 1 \
    --mem-fraction-static 0.6 \
    --max-running-requests 128 \
    --enable-memory-saver \
    --stream-interval 10 \
    --enable-mixed-chunk \
    --enable-weight-transfer-agent \
    --rollout-manager-address "http://${ROLLOUT_MANAGER_ADDR}"
