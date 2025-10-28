# To run PolyRL training pipeline

## To launch the training process

1. Config rollout manager
The configuration file of rollout manager is `src/rlboost/rollout-manager/config.toml`.
- `allowed_sender_ips` is for you to specify the range of ips for weight transfer from trainer to rollout instances
- `num_mooncake_groups_per_sender` is the max number of weight transfer groups per sender ip, each group will shard model weight into `num_mooncake_engines_per_group` and transfer weight in parallel to maximize bandwidth.

2. Launch an RL trainer

```bash
export GSM8K_DATA_DIR="path/to/gsm8k/data"
bash examples/scripts/run_async_grpo_pipeline.sh
```

It will automatically compile and launch the rollout manager and the weight transfer agent. The default port of the rollout manager is `5000`.

3. Launch a rollout instance.

Update the address in `launch_sglang.sh`
```bash
ROLLOUT_MANAGER_ADDR="<ROLLOUT_MANAGER_IP>:<PORT>" # Address of the head node of trainer
HOST_ADDR="<YOUR_HOST_IP>" # Address of the remote rollout engine, usually a spot instance
```

On each remote rollout engine, launch
```bash
bash examples/scripts/launch_sglang.sh
```

4. Run colocated RL baseline

```bash
bash examples/scripts/run_sync_grpo_default.sh
```

## To run trainer on multiple nodes

1. Start a ray cluster on the root node
```bash
ray start --head --dashboard-host=0.0.0.0
```

2. On other node, join the cluster by 
```bash
ray start --address='<follow the prompt on the root node>'
```

3. On the root node, start the job by
```bash
RAY_API_SERVER_ADDRESS='<follow the prompt on the root node>' ray job submit --working-dir . -- bash <your multi-node script>
```


