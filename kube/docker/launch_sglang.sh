#!/bin/bash

# Read from environment variables, using a default value if not set.
# This is more flexible than using positional arguments ($1, $2, etc.).
MODEL_NAME=${SGLANG_MODEL_NAME:-"Qwen/Qwen3-1.7B"}
HOST=${SGLANG_HOST:-$(hostname -i)}
PORT=${SGLANG_PORT:-"40000"}
MEM_FRAC=${SGLANG_MEM_FRAC:-"0.6"}
MAX_NUM_REQ=${SGLANG_MAX_REQ:-"128"}
TP_SIZE=${SGLANG_TP_SIZE:-2}

# --- Sanity Checks & User Feedback ---
echo "✅ Starting SGLang server with the following configuration:"
echo "   - Model: $MODEL_NAME"
echo "   - Host: $HOST"
echo "   - Port: $PORT"
echo "   - TP Size: $TP_SIZE"

# --- Launch the Server ---
python -m sglang.launch_server \
    --model-path "$MODEL_NAME" \
    --port "$PORT" \
    --host "$HOST" \
    --grammar-backend outlines \
    --tp-size "$TP_SIZE" \
    --mem-fraction-static "$MEM_FRAC" \
    --max-running-requests "$MAX_NUM_REQ" \
    --enable-memory-saver \
    --enable-mixed-chunk \
    --stream-output \
    --stream-interval 10