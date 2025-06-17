#!/bin/bash

# Set dataset directory - use environment variable GSM8K_DATA_DIR if set, otherwise use default
GSM8K_DATA_DIR=${GSM8K_DATA_DIR}
# Choose from sglang, sglang-disaggregated
ROLLOUT_NAME=${ROLLOUT_NAME:-"sglang-disaggregated"}

# Run PPO training
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=${GSM8K_DATA_DIR}/train.parquet \
 data.val_files=${GSM8K_DATA_DIR}/test.parquet \
 data.train_batch_size=64 \
 data.max_prompt_length=512 \
 data.max_response_length=4096 \
 actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
 actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=16 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen3-1.7B \
 critic.ppo_micro_batch_size_per_gpu=2 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console'] \
 trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.stream_fit=True \
 trainer.n_gpus_per_node=2 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=-1 \
 trainer.total_epochs=15 2>&1 | tee ppo_demo.log