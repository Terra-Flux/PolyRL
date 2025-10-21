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
The main entry point to run the PPO algorithm
"""

import datetime
import json
import logging
import os
import warnings
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import psutil
import torch
import torch.distributed
import torch.distributed as dist
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from peft import LoraConfig, TaskType, get_peft_model
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, Execute, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import hf_processor, hf_tokenizer, omega_conf_to_dataclass
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    is_cuda_available,
    is_npu_available,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    get_shard_placement_fn,
    init_fn,
    layered_summon_lora_params,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig, RolloutConfig
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class StreamActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, role, **kwargs)
        # polyrl-dev
        # NOTE(liuxs): track optimizer status
        if self._is_offload_optimizer:
            self._is_optimizer_loaded = False

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        rollout_name = self.config.rollout.name

        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = rollout_device_mesh["infer_tp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

        if rollout_name == "hf":
            raise NotImplementedError("hf rollout is not implemented for StreamActorRolloutRefWorker")

        elif rollout_name == "vllm":
            raise NotImplementedError("vllm rollout is not implemented for StreamActorRolloutRefWorker")

        elif rollout_name in ["sglang", 'sglang-disaggregated']:
            from verl.workers.rollout.sglang_rollout.sglang_rollout_remote import SGLangRolloutRemote
            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

            local_path = copy_to_local(self.config.model.path)
            # polyrl-dev
            # NOTE(liuxs): create the rollout wrapper for disaggregated mode
            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            if rollout_name == 'sglang-disaggregated':
                rollout = SGLangRolloutRemote(
                    actor_module=local_path,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    processing_class=self.processor if self.processor is not None else self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    trust_remote_code=trust_remote_code,
                )
                self.gen_iter = None
            else:
                raise ValueError("Use sglang-disaggregated instead of sglang to enable stream mode")
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            # polyrl-dev
            # rollout_manager_config = None if rollout_name != 'sglang-disaggregated' else self.config.rollout.rollout_manager
            # weight_sender_config_path = None if rollout_name != 'sglang-disaggregated' else rollout_manager_config.config_path
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                rollout_config=self.config.rollout,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                multi_stage_wake_up=self.config.rollout.multi_stage_wake_up,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)
        else:
            raise NotImplementedError(f"Rollout name: {self.config.rollout.name} is not supported")
            
        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import StreamDataParallelPPOActor, DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                # polyrl-dev
                self._is_optimizer_loaded = False
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        # polyrl-dev
        # use stream dp actor
        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = StreamDataParallelPPOActor(
                config=actor_cfg, actor_module=self.actor_module_fsdp, actor_optimizer=self.actor_optimizer
            )

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(
                trust_remote_code=self.config.model.get("trust_remote_code", False)
            )

        if self._is_ref:
            ref_model_path = self.config.model.path
            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)
            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if not self._is_actor and self._is_rollout:
            # If ActorRolloutRefWorker is initialized as a standalone rollout,
            # create a checkpoint manager for FSDP model to allow loading FSDP checkpoints for rollout.

            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update_stream")
    def update_actor_stream(self, data: DataProto):
        # polyrl-dev
        # load optimizer only when needed
        is_opt_step = data.meta_info.get("is_opt_step", False)
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer and is_opt_step and not self._is_optimizer_loaded:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())
            self._is_optimizer_loaded = True
        
        # polyrl-dev
        # update lr when specified
        is_lr_step = data.meta_info.get("is_lr_step", False)

        with self.ulysses_sharding_manager:
            # Support all hardwares
            data = data.to("cpu")  # data will to device with each micro batch on actor.update_policy
            # data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy_stream(data=data)
            # polyrl-dev
            # We need to compute this metric outside
            # global_num_tokens = data.meta_info["global_token_num"]
            # estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            # metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            # polyrl-dev
            # update lr when specified
            if is_lr_step:
                lr = self.actor_lr_scheduler.get_last_lr()[0]
                metrics["actor/lr"] = lr
                self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            # output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer and is_opt_step and is_lr_step and self._is_optimizer_loaded:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            self._is_optimizer_loaded = False
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output

    # polyrl-dev
    # update model weights cache in shared memory, transfer is handled by agent process
    @register(execute_mode=Execute.ALL)
    def update_weight_remote(self):
        self.rollout_sharding_manager.update_weight_remote()
        return True

    # polyrl-dev
    # NOTE(liuxs): all rollout engines must be resident during buffer zone, must execute on all ranks
    @register(execute_mode=Execute.ALL)
    def generate_sequences_remote(self, prompts: DataProto):
        # reset the gen_iter
        self.gen_iter = None

        # Support all hardwares
        prompts = prompts.to(get_torch_device().current_device())
        print(f"[Rank:{self.rank}] generate_sequences remote get called with dispatched {prompts.batch.size()} prompts")

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        # polyrl-dev
        # all ranks need to update but only rank0 submit the requests
        with self.rollout_sharding_manager: # only update local weights, remote weight is updated via update_weight_remote
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)
            if self.rank == 0:
                # NOTE(liuxs): only rank 0 will call this function 
                self.gen_iter = self.rollout.generate_sequences_remote(prompts=prompts) # rollout iterator will optimize return batch size
                status = next(self.gen_iter, False) # NOTE(liuxs): this means the local rollout has finished
            else:
                status = True
            dist.barrier() # Make sure local rollout engines exit simultaneously
            log_gpu_memory_usage("After submitting remote+local generation", logger=logger)
            # NOTE(liuxs): the memory of local engine will be released once exit this scope, 
            # make sure the local rollout is done
        log_gpu_memory_usage("After exiting local generation", logger=logger)
        print("all requests submitted")
        return status
    
    @register(execute_mode=Execute.RANK_ZERO)
    def get_stream_batches(self):
        # polyrl-dev
        # because iterator is not serializable, we need to get the microbatch from the iterator
        # and return it to the trainer
        if self.gen_iter is None:
            raise ValueError("gen_iter is not initialized")
        batch = next(self.gen_iter, None)
        if batch is None:
            return None
        return batch
    

# FIXME(liuxs): StreamCriticWorker is not fully tested yet, use GRPO for better stability
class StreamCriticWorker(CriticWorker):
    def __init__(self, config: FSDPCriticConfig):
        super().__init__(config)
        
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from verl.workers.critic import StreamDataParallelPPOCritic

        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config
        )

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
            log_gpu_memory_usage("After offload critic model during init", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
            log_gpu_memory_usage("After offload critic optimizer during init", logger=logger)

        self.critic = StreamDataParallelPPOCritic(
            config=self.config, critic_module=self.critic_module, critic_optimizer=self.critic_optimizer
        )

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.critic_module,
            optimizer=self.critic_optimizer,
            lr_scheduler=self.critic_lr_scheduler,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            checkpoint_config=self.config.checkpoint,
        )
    
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="pink")
    def update_critic_stream(self, data: DataProto):
        # Support all hardwares
        data = data.to(get_device_id())
        
        # polyrl-dev
        # load optimizer when needed
        is_opt_step = data.meta_info.get("is_opt_step", False)
        
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer and is_opt_step:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=get_device_id())
        
        # polyrl-dev
        # update lr when specified
        is_lr_step = data.meta_info.get("is_lr_step", False)

        # perform forward computation
        with self.ulysses_sharding_manager:
            # data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name="update_critic", logger=None) as timer:
                metrics = self.critic.update_critic_stream(data=data)
            # delta_time = timer.last
            
            # polyrl-dev
            # calculate flops outside
            # global_num_tokens = data.meta_info["global_token_num"]
            # estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            # metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            # polyrl-dev
            # update lr when specified
            if is_lr_step:
                self.critic_lr_scheduler.step()
                lr = self.critic_lr_scheduler.get_last_lr()[0]
                metrics["critic/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            # output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer and is_opt_step:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        output = output.to("cpu")
        return output
