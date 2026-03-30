# NOTE: 
# 1. memory fraction is set to 0.6, 0.9 will cause OOM because of the large amount of meta data through requests
TP_SIZE="${1:-8}"

# Standard port alignment with bench.sh
PORT=30000

# Clear restrictive environment variables that cause HAL initialization errors
unset TPU_VISIBLE_DEVICES
unset TPU_CHIPS_PER_HOST_BOUNDS
unset TPU_HOST_BOUNDS

DEVICE_INDEX_ARG=""
if [ "$TP_SIZE" -eq 2 ]; then
    DEVICE_INDEX_ARG="--device-indexes 0 1"
elif [ "$TP_SIZE" -eq 4 ]; then
    DEVICE_INDEX_ARG="--device-indexes 0 1 2 3"
elif [ "$TP_SIZE" -eq 8 ]; then
    DEVICE_INDEX_ARG=""
else
    echo "TP_SIZE $TP_SIZE not supported for explicit mesh setting"
    exit 1
fi

# 3. (Optional) Set a unique ID/Port to avoid conflicts if running multiple jobs
export TPU_MESH_CONTROLLER_ADDRESS=localhost:8476
export TPU_MESH_CONTROLLER_PORT=8476

export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
# export JAX_PLATFORMS=tpu

uv run python -u -m sgl_jax.launch_server \
    --model-path Qwen/Qwen3-8B \
    --trust-remote-code  \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1  \
    --tp-size=$TP_SIZE \
    --device=tpu \
    --node-rank=0 \
    --mem-fraction-static=0.8 \
    --max-prefill-tokens=32768 \
    --chunked-prefill-size=8192 \
    --context-len 32768 \
    --download-dir=/tmp \
    --log-requests-level=0 \
    --dtype="auto"  \
    --host "0.0.0.0" \
    --port $PORT \
    --page-size 128 \
    --max-running-requests=512 \
    $DEVICE_INDEX_ARG
