# Instructions

## 0. Launch SGLang server
On GPU node, just run native SGLang serving using command like,
```bash
uv run python -u -m sglang.launch_server \
    --trust-remote-code  \
    --dist-init-addr=0.0.0.0:10011 \
    --model-path Qwen/Qwen3-8B \
    --host "0.0.0.0" \
    --port 30000 \
    --tp-size 2 \
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
```

On TPU node, launch through the patch, which will enable update weights through API
```bash
uv run python -u sglang-patch/launch_server.py \
    --model-path Qwen/Qwen3-8B \
    --trust-remote-code  \
    --dist-init-addr=0.0.0.0:10011 \
    --nnodes=1  \
    --tp-size=4 \
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
    --port 30000 \
    --page-size 128 \
    --max-running-requests=512 \
    --device-indexes 0 1 2 3
```


## 1. Launch dummy sender
A dummy sender will simulate a manager + a trainer + a sender proxy.

The process is:
1. The sender proxy will establish multiple engines to fully utilize the TCP bandwidth
2. The manager will handle weight receiver registration and assign weight transfer engines to the receiver.
3. Trainer will periodically write weights into the shm, and let agent handle the weight transfer. 
4. Sender agent will push weight to its assigned receivers and notify rollout manager.
5. Rollout manager then notify all receivers to proceed receiving and sharding and update weights.
6. Back to 3.

Launch the dummy sender via
```bash
python3 weight-transfer/dummy_sender.py --interval 30 --model Qwen/Qwen3-8B --manager-port 8000 
```

If sender fails 5 times handshake with registered agent, the agent will be considered as dead.

## 2. Launch weight receiver agents
On GPU node, launch weight receiver agent specifying the tp size,
```bash
python launch_receiver_gpu.py --tp-size 4 \
    --manager-url "http://10.202.15.230:8000" \
    --sglang-url "http://127.0.0.1:30000"
```

On TPU node, launch weight receiver agent directly because it writes into shm,
```bash
python launch_receiver_tpu.py \
    --manager-url "http://10.202.15.230:8000" \
    --sglang-url "http://127.0.0.1:30000"
```

The receiver agent will detect rollout engine via `health/` and even if the rollout engine fails, the agent keeps running. It will reconnect once the rollout engine restart.


# Known Issues
- You MAY encounter `ValueError: token_to_kv_pool_allocator memory leak detected! ...` error when calling `get_server_info/`, if the weight is updated when the engine is generating. Still debugging.
