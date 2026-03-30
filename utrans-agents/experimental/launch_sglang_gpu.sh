#!/bin/bash

# NOTE: 
# 1. memory fraction is set to 0.6, 0.9 will cause OOM because of the large amount of meta data through requests
TP_SIZE="$1"

uv run python -u -m sglang.launch_server \
    --trust-remote-code  \
    --dist-init-addr=0.0.0.0:10011 \
    --model-path Qwen/Qwen3-8B \
    --host "0.0.0.0" \
    --port 30000 \
    --tp-size $TP_SIZE \
    --mem-fraction-static 0.8 \
    --max-running-requests 512 \
    --stream-interval 32 \
    --enable-mixed-chunk \
    --max-prefill-tokens=32768 \
    --chunked-prefill-size=8192 \
    --enable-mixed-chunk \
    --context-len 32768 \
    --stream-output \
    --log-requests-level=0 \
    --cuda-graph-max-bs 512 \
    --dtype="auto"

