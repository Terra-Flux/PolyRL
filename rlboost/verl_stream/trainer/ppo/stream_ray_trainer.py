# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (RayClassWithInitArgs, RayResourcePool,
                                        RayWorkerGroup)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role, compute_advantage, apply_kl_penalty, WorkerType

# polyrl-dev
import requests   
from rlboost.verl_stream.trainer.ppo.reward import compute_reward, compute_reward_async

class StreamRayPPOTrainer(RayPPOTrainer):
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """
        super().__init__(config,
                         tokenizer,
                         role_worker_mapping,
                         resource_pool_manager,
                         ray_worker_group_cls,
                         processor,
                         reward_fn,
                         val_reward_fn,
                         train_dataset,
                         val_dataset,
                         collate_fn,
                         train_sampler,
                         device_name)
        
    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        # polyrl-dev
        # The RayWorkerGroup created by wg_dicts is the reference of the workers created by RayWorkerGroup
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            assert (
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                is not None
            ), "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            # polyrl-dev
            # By default, all workers share a global pool
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )
    
    def generate_sequences_remote(self, batch: DataProto):
        """
        Hybrid local and remote generation
        Local generation is synchronous
        Remote generation is asynchronous
        """
        # polyrl-dev
        # compute microbatch size
        n_gpus = self.resource_pool_manager.get_n_gpus()
        # NOTE(liuxs): ppo_mini_batch_size is actually ppo_mini_batch_size per GPU
        mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * n_gpus 
        if not self.config.actor_rollout_ref.actor.use_dynamic_bsz:
            micro_batch_size = self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * n_gpus
            print(f"Global minibatch size {mini_batch_size} and global microbatch size {micro_batch_size}")
        else:
            print(f"Global minibatch size {mini_batch_size} and with dynamic microbatch size")
                    
        # Calculate number of mini batches and split the entire batch
        real_train_batch_size = batch.batch.batch_size[0] * self.config.actor_rollout_ref.rollout.n
        num_mini_batches = real_train_batch_size // mini_batch_size
        real_mini_batch_size = [mini_batch_size for _ in range(num_mini_batches)]
        cum_mini_batch_size = [mini_batch_size * (i+1) for i in range(num_mini_batches)]
        # add residual data
        res_batch_size = real_train_batch_size % mini_batch_size
        if res_batch_size != 0:
            real_mini_batch_size.append(res_batch_size)
            cum_mini_batch_size.append(cum_mini_batch_size[-1] + res_batch_size)

        # polyrl-dev
        # Local rollout engine is also a server managed by rollout-manager
        # meta data will be processed inside the rollout iterator, just pass the whole batch
        # call RANK_ZERO to avoid sharding the batch
        status = self.actor_rollout_wg.generate_sequences_remote(batch)
        # status is a list equal to number of GPUs
        if not status[0]:
            raise ValueError("generate_sequences_remote failed")
            
        # NOTE(liuxs): return a list of minibatch_size
        print(f"{real_mini_batch_size=}, {cum_mini_batch_size=}")
        return cum_mini_batch_size

    # polyrl-dev 
    # create a new stream version of fit function for stream training
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        from verl.utils.py_functional import append_to_dict

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
            
        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            raise NotImplementedError(f"Async rollout is not tested on stream fit yet!")
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False
        
        # polyrl-dev
        # update remote weight at the beginning of the training
        if not self.actor_rollout_wg.update_weight_remote():
            raise ValueError("update weight remote failed")
        
        # polyrl-dev
        num_rollout_instances = 0
        local_gen_s = 0

        for epoch in range(self.config.trainer.total_epochs):
            # polyrl-dev
            # Batch size sampled for one training iteration of different RL algorithms.
            for batch_dict in self.train_dataloader:
                # polyrl-dev
                # load a training batch and send to rollout manager
                metrics = {}
                timing_raw = {}
                
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                
                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                
                batch.meta_info["global_steps"] = self.global_steps
                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # submit the batch
                    with marked_timer("local_gen", timing_raw):
                        cum_minibatch_size_list = self.generate_sequences_remote(batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        raise NotImplementedError("REMAX is not tested in stream mode")
                    
                    # Process each minibatch
                    all_ibatch_results = []
                    
                    # polyrl-dev
                    # create a step progress bar as we are streaming results
                    step_progress_bar = tqdm(total=cum_minibatch_size_list[-1], desc="Step Progress", colour="red")
                    
                    # polyrl-dev
                    # NOTE(liuxs): except from critic and actor update, all the other ops process as much requests as possible 
                    curr_minibatch = 0 # update every optimizer step
                    processed_responses = 0 # update after each actor update
                    while True:
                        with marked_timer("gen", timing_raw):
                            ibatch = self.actor_rollout_wg.get_stream_batches() # return a ibatch each time
                        if ibatch == None:
                            print(f"All Rollout results have been processed!")
                            break
                        
                        if "response_mask" not in ibatch.batch.keys():
                            raise KeyError(f"response_mask should have been calculated in post-processing")
                            batch.batch["response_mask"] = compute_response_mask(batch)
                        # Balance the number of valid tokens across DP ranks.
                        # NOTE: This usually changes the order of data in the `batch`,
                        # which won't affect the advantage calculation (since it's based on uid),
                        # but might affect the loss calculation (due to the change of mini-batching).
                        # TODO: Decouple the DP balancing and mini-batching.
                        if self.config.trainer.balance_batch:
                            # FIXME(liuxs): .tolist was removed inside _balance_batch by mistake, 
                            # which will add a tensor object to the metrics,
                            # we have to skip it for now
                            self._balance_batch(ibatch, metrics={})
                                                
                        with marked_timer("reward", timing_raw, color="yellow"):
                            # compute reward model score
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(ibatch)
                                ibatch = ibatch.union(reward_tensor)

                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(data=ibatch, reward_fn=self.reward_fn)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(ibatch, self.reward_fn)

                        # polyrl-dev
                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            # polyrl-dev
                            # e.g., old_log_probs: Tensor(shape=torch.Size([256, 256]), 
                            #   device=cpu, dtype=torch.float32, is_shared=False), 
                            # 256 is batch size, 256 is response length
                            # old_log_prob -> (bs * n, seqlen)
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(ibatch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = ibatch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            ibatch = ibatch.union(old_log_prob)

                            if "rollout_log_probs" in ibatch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(ibatch))

                        if self.use_reference_policy:
                            # compute reference log_prob
                            # polyrl-dev
                            # ref_log_prob -> (bs * n, seqlen)
                            with marked_timer("ref", timing_raw, color="olive"):
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(ibatch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(ibatch)
                                ibatch = ibatch.union(ref_log_prob)

                        # compute values
                        # polyrl-dev
                        if self.use_critic:
                            with marked_timer("values", timing_raw, color="cyan"):
                                values = self.critic_wg.compute_values(ibatch)
                                ibatch = ibatch.union(values)

                        with marked_timer("adv", timing_raw, color="brown"):
                            # we combine with rule-based rm
                            reward_extra_infos_dict: dict[str, list]
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            ibatch.batch["token_level_scores"] = reward_tensor

                            if reward_extra_infos_dict:
                                ibatch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # compute rewards. apply_kl_penalty if available
                            if self.config.algorithm.use_kl_in_reward:
                                # polyrl-dev
                                ibatch, kl_metrics = apply_kl_penalty(
                                    ibatch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                ibatch.batch["token_level_rewards"] = ibatch.batch["token_level_scores"]

                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            )  # GRPO adv normalization factor

                            ibatch = compute_advantage(
                                ibatch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                config=self.config.algorithm,
                            )
                        
                        # Calculate number of micro batches and split the entire ibatch
                        ibatch_size = ibatch.batch.batch_size[0]
                        
                        acc_responses = processed_responses + ibatch_size # global index of this ibatch
                        response_ibatch_idx = 0 # index within this ibatch
                        while acc_responses >= cum_minibatch_size_list[curr_minibatch]:
                            # NOTE(liuxs): able to proceed optimizer step
                            to_process_responses = cum_minibatch_size_list[curr_minibatch] - processed_responses
                            to_process_batch = ibatch[response_ibatch_idx:response_ibatch_idx + to_process_responses]
                            assert to_process_batch.batch.size()[0] == to_process_responses, \
                                f"to process batch size {to_process_batch.batch.size()[0]} != {to_process_responses}"
                            to_process_batch.meta_info["is_opt_step"] = True
                            
                            if curr_minibatch == len(cum_minibatch_size_list) - 1:
                                # process the last minibatch, need to update lr
                                to_process_batch.meta_info["is_lr_step"] = True
                            if self.use_critic:
                                # fwd bwd critic at microbatch level (accumulate gradients)
                                with marked_timer("update_critic", timing_raw, color="pink"):
                                    critic_output = self.critic_wg.update_critic_stream(to_process_batch)
                                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                                    append_to_dict(metrics, critic_output_metrics)

                            if self.config.trainer.critic_warmup <= self.global_steps:
                                # fwd bwd actor at microbatch level (accumulate gradients)
                                with marked_timer("update_actor", timing_raw, color="red"):
                                    # polyrl-dev
                                    # TODO(liuxs): I am not sure if we really need this multi-turn
                                    to_process_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                    actor_output = self.actor_rollout_wg.update_actor_stream(to_process_batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                append_to_dict(metrics, actor_output_metrics)
                                
                            # NOTE(liuxs): update counter
                            response_ibatch_idx += to_process_responses
                            processed_responses += to_process_responses
                            curr_minibatch += 1
                            step_progress_bar.update(to_process_responses)
                            if curr_minibatch == len(cum_minibatch_size_list):
                                print(f"All {curr_minibatch} microbatches are processed")
                                break
                        
                        # NOTE(liuxs): process reminder ibatch if necessary
                        to_process_batch = ibatch[response_ibatch_idx:]
                        to_process_responses = to_process_batch.batch.size()[0]
                        if to_process_responses:
                            # Process responses without optimizer step
                            if self.use_critic:
                                # fwd bwd critic at microbatch level (accumulate gradients)
                                with marked_timer("update_critic", timing_raw, color="pink"):
                                    critic_output = self.critic_wg.update_critic_stream(to_process_batch)
                                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                                    append_to_dict(metrics, critic_output_metrics)
                                    
                            if self.config.trainer.critic_warmup <= self.global_steps:
                                # fwd bwd actor at microbatch level (accumulate gradients)
                                with marked_timer("update_actor", timing_raw, color="red"):
                                    # polyrl-dev
                                    # TODO(liuxs): I am not sure if we really need this
                                    to_process_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                    actor_output = self.actor_rollout_wg.update_actor_stream(to_process_batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                append_to_dict(metrics, actor_output_metrics)
                                
                        processed_responses += to_process_responses

                        # update counter
                        all_ibatch_results.append(ibatch)
                        step_progress_bar.update(to_process_responses)
                    
                    # polyrl-dev
                    with marked_timer("update_weight", timing_raw):
                        # polyrl-dev
                        # update weight buffer cache
                        if not self.actor_rollout_wg.update_weight_remote():
                            raise ValueError("update weight remote failed")
                    
                    # polyrl-dev
                    # clear step progress bar
                    step_progress_bar.close()
                    
                    # Merge all minibatch results back into full batch for logging
                    batch = DataProto.concat(all_ibatch_results)
                    
                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # polyrl-dev
                    # NOTE(liuxs): I think we actually don't need this feature but I'll keep it
                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                # polyrl-dev
                # compute global_token_num for the entire batch
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                
                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile
                    
                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                # polyrl-dev
                # NOTE: reduce the accumulated metrics - convert lists to single values
                metrics = reduce_metrics(metrics)
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics["training/num_rollout_instances"] = num_rollout_instances
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                metrics["perf/throughput_all_gpus"] = metrics["perf/throughput"] * n_gpus

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    raise NotImplementedError(f"AbstractCurriculumSampler is not tested on stream mode!")
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                remote_metrics = self.update_metrics(metrics)
                num_rollout_instances = int(remote_metrics["new_num_rollout_instances"])
                local_gen_s = int(remote_metrics["new_max_gen_s"])
                
                progress_bar.update(1)
                self.global_steps += 1
                
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

    def update_metrics(self, metrics: dict):
        
        rollout_mgr_url = self.config.actor_rollout_ref.rollout.rollout_manager.endpoint
        assert rollout_mgr_url is not None, "rollout_manager.endpoint must be set"

        payload = {
            "step_time_s": int(metrics.get("timing_s/step", 0)),
            "trainer_bubble_time_s": int(metrics.get("timing_s/gen", 0)),
            "step_throughput": int(metrics.get("perf/throughput", 0)),
        }
        with requests.post(f"{rollout_mgr_url.rstrip('/')}/update_metrics", json=payload) as resp:
            resp.raise_for_status()
            
            return resp.json()
        


