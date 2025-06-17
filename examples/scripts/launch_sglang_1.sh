#!/bin/bash

# NOTE: 
# 1. memory fraction is set to 0.6, 0.9 will cause OOM because of the large amount of meta data through requests
ROLLOUT_MANAGER_ADDRESS=${ROLLOUT_MANAGER_ADDRESS:-"localhost:5000"}

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-1.7B \
    --port 40000 \
    --host localhost \
    --mooncake-handshake-port 21000 \
    --grammar-backend outlines \
    --tp-size 1 \
    --mem-fraction-static 0.6 \
    --enable-memory-saver \
    --enable-weight-transfer-agent \
    --rollout-manager-address "http://$ROLLOUT_MANAGER_ADDRESS"