HOST="0.0.0.0"
PORT=30000
RAND_RATE=0.0 # 0.0 is fully random on sgl_bench_serving

python3 -m sglang.bench_serving \
    --backend sglang \
    --base-url http://$HOST:$PORT \
    --request-rate inf \
    --dataset-name random \
    --flush-cache \
    --num-prompts 2048 \
    --random-input-len 2048 \
    --random-output-len 128 \
    --max-concurrency 512 \
    --random-range-ratio $RAND_RATE \
    --warmup-requests 1 
