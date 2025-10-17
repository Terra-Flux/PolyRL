# To run PolyRL training pipeline

# To launch the training process

1. Launch an RL trainer

Update `SENDER_IPS` in `run_async_grpo_pipeline.sh`. It means the network interfaces that you want to use for weight transfer.

```bash
export GSM8K_DATA_DIR="path/to/gsm8k/data"
bash examples/scripts/run_async_grpo_pipeline.sh
```

It will automatically compile and launch the rollout manager and the weight transfer agent. The default port of the rollout manager is `5000`.

2. Launch a rollout instance.

Update the address in `launch_sglang.sh`
```bash
ROLLOUT_MANAGER_ADDR="<ROLLOUT_MANAGER_IP>:<PORT>" # Address of the head node of trainer
HOST_ADDR="<YOUR_HOST_IP>" # Address of the remote rollout engine, usually a spot instance
```

On each remote rollout engine, launch
```bash
bash examples/scripts/launch_sglang.sh
```

3. Run colocated RL baseline

```bash
bash examples/scripts/run_sync_grpo_default.sh
```
