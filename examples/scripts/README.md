# To run PolyRL training pipeline

# NOTE:

1. Make sure `port` is different for each rollout instance.
2. Make sure the `mooncake_handshake_port` is far enough for each rollout instance.


# To launch the training process

1. Launch an RL trainer

```bash
export GSM8K_DATA_DIR="path/to/gsm8k/data"
bash examples/scripts/run_async_ppo_pipeline.sh
```

It will automatically compile and launch the rollout manager and the weight transfer agent. The default port of the rollout manager is `5000`.

When the weight transfer agent is ready, it will print
```
Waiting for rollout instances to register... 
```
and you can proceed to launch rollout instances.

2. Launch a rollout instance.

```bash
export CUDA_VISIBLE_DEVICES=6,7 # make sure it is exclusive to the trainer
bash examples/scripts/launch_sglang_1.sh
```

It assumes the rollout manager is running on the same machine by default. If not, you need to set the `ROLLOUT_MANAGER_ADDRESS` to the address of the rollout manager (e.g., `ROLLOUT_MANAGER_ADDRESS=1.2.3.4:5000`).

3. Add additional rollout instances

```bash
export CUDA_VISIBLE_DEVICES=8,9 # make sure it is exclusive to the trainer and the first one
bash examples/scripts/launch_sglang_2.sh
```

This new instance will be added to a queue and join the rollout from the next training step.

