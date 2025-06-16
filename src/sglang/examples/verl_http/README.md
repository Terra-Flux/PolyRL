# To test the VerlHttpEngine

1. Start the VerlHttpEngine server:

```bash
./launch_sglang.sh
```

2. Run the HTTP rollout script:

```bash
python run_http_rollout.py --model_index 1 --batch_size 4 --max_new_tokens 100 --temperature 0.8 --use_chat_template
```

