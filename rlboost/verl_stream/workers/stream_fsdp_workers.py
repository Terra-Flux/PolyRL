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

import asyncio
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
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, Execute, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    collect_lora_params,
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
    replace_lora_wrapper,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import compute_position_id_with_mask, convert_weight_keys
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.py_functional import convert_to_regular_types
from verl.workers.config import FSDPCriticConfig, FSDPEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.rollout import get_rollout_class
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

# polyrl-dev
from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
from rlboost.verl_stream.workers.config import FSDPCriticConfig, FSDPEngineConfig, RolloutConfig, HFModelConfig
from rlboost.weight_transfer.fsdp_interface import FSDPInterface

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

        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout, dataclass_type=RolloutConfig)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config

        # 2. build rollout device mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )
        rollout_name = self.config.rollout.name

        if rollout_name == "hf":
            raise NotImplementedError("hf rollout is not implemented for StreamActorRolloutRefWorker")

        elif rollout_name == "vllm":
            raise NotImplementedError("vllm rollout is not implemented for StreamActorRolloutRefWorker")
        
        elif rollout_name == "sglang":
            raise NotImplementedError("sglang rollout is not implemented for StreamActorRolloutRefWorker, please use sglang-disaggregated")
        else:
            is_collect = (
                rollout_device_mesh["infer_tp"].get_local_rank() == 0
                and rollout_device_mesh["infer_pp"].get_local_rank() == 0
            )
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        # 3. init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build rollout model
        from rlboost.verl_stream.workers.rollout.sglang_rollout.sglang_rollout_remote import SGLangRolloutRemote

        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = SGLangRolloutRemote(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

        # Full params
        if torch.distributed.get_world_size() == 1 and fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(),
            )
        elif fsdp_version(self.actor_module_fsdp) == 1:
            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
        
        # polyrl-dev
        if self._is_actor:
            local_world_size = int(os.environ.get("RAY_LOCAL_WORLD_SIZE", 1))
            # FIXME: RAY_LOCAL_RANK is not set by default in verl/single_controller/ray/base.py, use mod to calculate the local rank
            local_rank = self.rank % local_world_size
            print(f"Local rank of {self.rank} is {local_rank} / {local_world_size}")
            self.weight_transfer = FSDPInterface(
                local_rank=local_rank,
                global_rank=self.rank,
                params=self.actor_module_fsdp.state_dict(),
                rollout_manager_endpoint=self.config.rollout.rollout_manager.endpoint,
                weight_sender_config_path=self.config.rollout.rollout_manager.config_path,
            )

        # used for LoRA
        self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
        self.layered_summon = self.config.rollout.get("layered_summon", False)

        # 5. switch to trainer mode
        # NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
        # For sync mode, we directly switch to trainer mode here.
        # For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
        if rollout_config.mode == "sync" and self._is_actor:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.trainer_mode())


    async def update_weight_cache(self):
        """Update model weight cache for pull-based transfer engine."""
        aggressive_empty_cache(force_sync=True)

        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "peft_config"):  # LoRA
            # FIXME(liuxs): test on lora
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.config.rollout.get("layered_summon", False),
                base_sync_done=self.base_sync_done,
            )
            if not self.base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            params = self.actor_module_fsdp.state_dict()

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )
        
        # Copy, not share memory
        # polyrl-dev
        # make sure all nodes are ready before weight copy
        # FIXME(liuxs): only add barriers for actors
        if dist.is_initialized():
            dist.barrier()
        self.weight_transfer.update_weights_with_agent(params)
        # make sure all nodes are ready after weight copy
        if dist.is_initialized():
            dist.barrier()
        log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)
        
        # polyrl-dev 
        # skip handling base model on lora, delete model params
        del params
        
        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)            
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor
        from rlboost.verl_stream.workers.actor import StreamDataParallelPPOActor

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
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

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
        if self._is_actor:  
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(self.update_weight_cache())
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
        if self._is_actor:  # For rollout only, we do not switch context.
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)
        
        # FIXME(liuxs): only sync rollout group
        dist.barrier() # Make sure all rollout engines switched context

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
        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

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

        from rlboost.verl_stream.workers.critic import StreamDataParallelPPOCritic

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
