#!/bin/bash

ulimit -n 65536  # Increase max open files

# Set dataset directory - use environment variable GSM8K_DATA_DIR if set, otherwise use default
GSM8K_DATA_DIR=${GSM8K_DATA_DIR}
# Choose from sglang, sglang-disaggregated
ROLLOUT_NAME=${ROLLOUT_NAME:-"sglang-disaggregated"}

# Run GRPO training
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=${GSM8K_DATA_DIR}/train.parquet \
 data.val_files=${GSM8K_DATA_DIR}/test.parquet \
 data.train_batch_size=64 \
 data.max_prompt_length=512 \
 data.max_response_length=4096 \
 data.filter_overlong_prompts=True \
 data.truncation='error' \
 actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
 actor_rollout_ref.model.path=Qwen/Qwen3-1.7B \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
 actor_rollout_ref.rollout.n=8 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
 algorithm.use_kl_in_reward=False \
 trainer.logger=['console','wandb'] \
 trainer.project_name='grpo_example_gsm8k' \
 trainer.experiment_name='qwen3_1.7b_gsm8k_async_grpo' \
 trainer.val_before_train=False \
 trainer.critic_warmup=0 \
 trainer.n_gpus_per_node=2 \
 trainer.stream_fit=True \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=-1 \
 trainer.total_epochs=15 2>&1 | tee grpo_demo.log