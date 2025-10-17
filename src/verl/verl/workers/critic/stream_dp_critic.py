# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Implement a multiprocess PPOCritic
"""

import logging
import os

import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic
from verl.workers.critic.dp_critic import DataParallelPPOCritic

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# polyrl-dev
# stream version of critic worker
class StreamDataParallelPPOCritic(DataParallelPPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config, critic_module, critic_optimizer)
        # polyrl-dev
        # reset optimizer
        if self.critic_optimizer:
            self.critic_optimizer.zero_grad()
            
                
    # def _forward_micro_batch = super()._forward_micro_batch

    # def _optimizer_step = super()._optimizer_step

    # def compute_log_prob = super().compute_log_prob

    # def update_critic = super().update_critic
    
    # polyrl-dev
    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic_stream(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}
        is_opt_step = data.meta_info.get("is_opt_step", False)

        select_keys = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids", "values", "returns"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        ibatch = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # polyrl-dev
        # already in sub-minibatch, no need for further split
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
        # NOTE(liuxs): zero_grad is moved to init, because this function will be called multiple times
        # self.critic_optimizer.zero_grad()

        for micro_batch in micro_batches:
            micro_batch_metrics = {}
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            response_mask = model_inputs["response_mask"]
            values = model_inputs["values"]
            returns = model_inputs["returns"]

            vpreds = self._forward_micro_batch(model_inputs)
            vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                vpreds=vpreds,
                values=values,
                returns=returns,
                response_mask=response_mask,
                cliprange_value=self.config.cliprange_value,
                loss_agg_mode=self.config.loss_agg_mode,
            )
            if self.config.use_dynamic_bsz:
                # relative to the dynamic bsz
                loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                loss = vf_loss * loss_scale_factor
            else:
                loss_scale_factor = 1 / self.gradient_accumulation
                loss = vf_loss * loss_scale_factor

            loss.backward()

            micro_batch_metrics.update(
                {
                    "critic/vf_loss": vf_loss.detach().item() * loss_scale_factor,
                    "critic/vf_clipfrac": vf_clipfrac.detach().item(),
                    "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
                }
            )

            append_to_dict(metrics, micro_batch_metrics)

        # polyrl-dev
        # optimizer step only for the last microbatch
        if is_opt_step:
            grad_norm = self._optimizer_step()
            mini_batch_metrics = {"critic/grad_norm": grad_norm.detach().item()}
            append_to_dict(metrics, mini_batch_metrics)
            self.critic_optimizer.zero_grad()
        return metrics
