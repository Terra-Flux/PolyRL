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
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import (index_first_axis, pad_input,
                                         rearrange, unpad_input)
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import (
        index_first_axis, pad_input, rearrange, unpad_input)
    
from verl.workers.actor.dp_actor import DataParallelPPOActor


__all__ = ["StreamDataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# polyrl-dev
# stream version of actor worker
class StreamDataParallelPPOActor(DataParallelPPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config, actor_module, actor_optimizer)
        
        # polyrl-dev
        # reset optimizer
        if self.actor_optimizer:
            self.actor_optimizer.zero_grad()
            
    # def _forward_micro_batch = super()._forward_micro_batch

    # def _optimizer_step = super()._optimizer_step

    # def compute_log_prob = super().compute_log_prob

    # def update_policy = super().update_policy
    
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy_stream(self, data: DataProto):
        """Update policy model in stream mode.

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.
                
                and meta_info including
                
                ``is_opt_step``: bool. If True, run optimizer step after the last backward

        Returns:
            metrics (dict) 
        """
        # make sure we are in training mode
        self.actor_module.train()
        is_opt_step = data.meta_info.get("is_opt_step", False)

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        
        # polyrl-dev
        # No need to split into minibatches as we've already in sub-minibatch scale
        ibatch = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        if self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, _ = prepare_dynamic_batch(ibatch, max_token_len=max_token_len)
        else:
            self.gradient_accumulation = (
                self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            )
            micro_batches = ibatch.split(self.config.ppo_micro_batch_size_per_gpu)

        # polyrl-dev
        # the input is already a sub-minibatch, so just split into microbatches
        if self.config.ppo_epochs != 1:
            raise NotImplementedError(f"{self.config.ppo_epochs=} != 1 is not implemented in stream model yet!")
        # NOTE(liuxs): zero_grad() is moved to init, because this function will be called multiple times
        # self.actor_optimizer.zero_grad()
        metrics = {}
        # polyrl-dev
        # Forward/backward at the microbatch level, accumulate gradient
        # Update model at the minibatch level
        for micro_batch in micro_batches:
            # Support all hardwares
            micro_batch = micro_batch.to(get_device_id())
            micro_batch_metrics = {}
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            response_mask = model_inputs["response_mask"]
            old_log_prob = model_inputs["old_log_probs"]
            advantages = model_inputs["advantages"]
            
            entropy_coeff = self.config.entropy_coeff
            loss_agg_mode = self.config.loss_agg_mode

            if self.config.use_dynamic_bsz:
                loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
            else:
                loss_scale_factor = 1 / self.gradient_accumulation

            # all return: (bsz, response_length)
            calculate_entropy = False
            if entropy_coeff != 0:
                calculate_entropy = True
            entropy, log_prob = self._forward_micro_batch(
                model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
            )

            loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
            # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
            # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
            # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
            policy_loss_fn = get_policy_loss_fn(loss_mode)
            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                response_mask=response_mask,
                loss_agg_mode=loss_agg_mode,
                config=self.config,
            )

            if entropy_coeff != 0:
                entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff
            else:
                policy_loss = pg_loss

            if self.config.use_kl_loss:
                ref_log_prob = model_inputs["ref_log_prob"]
                # compute kl loss
                kld = kl_penalty(
                    logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                )
                kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

            if self.config.use_dynamic_bsz:
                # relative to the dynamic bsz
                loss = policy_loss * loss_scale_factor
            else:
                loss = policy_loss * loss_scale_factor
            loss.backward()
            
            micro_batch_metrics.update(
                {
                    "actor/pg_loss": pg_loss.detach().item(),
                    "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                    "actor/ppo_kl": ppo_kl.detach().item(),
                    "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                }
            )
            append_to_dict(metrics, micro_batch_metrics)

        if is_opt_step:
            grad_norm = self._optimizer_step()
            mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, mini_batch_metrics)
            self.actor_optimizer.zero_grad()
        return metrics
