#!/bin/bash

# NOTE: 
# 1. memory fraction is set to 0.6, 0.9 will cause OOM because of the large amount of meta data through requests

python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 40000 \
    --host 0.0.0.0 \
    --grammar-backend outlines \
    --tp-size 2 \
    --mem-fraction-static 0.6 \
    --enable-memory-saver
    