#!/bin/bash

# NOTE: 
# 1. memory fraction is set to 0.6, 0.9 will cause OOM because of the large amount of meta data through requests

python -m sglang.launch_server \
    --model-path Qwen/Qwen3-1.7B \
    --port 40000 \
    --host localhost \
    --grammar-backend outlines \
    --tp-size 1 \
    --mem-fraction-static 0.6 \
    --max-running-requests 128 \
    --enable-memory-saver \
