#!/bin/bash

# host is either $1 or 0.0.0.0
HOST=${1:-0.0.0.0}

# port is either $2 or 30000
PORT=${2:-30000}

# tp-size is either $3 or 1
TP_SIZE=${3:-1}

# NOTE: 
# 1. memory fraction is set to 0.6, 0.9 will cause OOM because of the large amount of meta data through requests

echo "Launching SGLang server with:
    HOST: $HOST
    PORT: $PORT
    TP_SIZE: $TP_SIZE
"

python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --port $PORT \
    --host $HOST \
    --grammar-backend outlines \
    --tp-size $TP_SIZE \
    --mem-fraction-static 0.6 \
    --enable-memory-saver \
    --enable-mixed-chunk \
    --max-running-requests 64 \

    # --decode-log-interval 1 \
    # --max-total-tokens 20480

    