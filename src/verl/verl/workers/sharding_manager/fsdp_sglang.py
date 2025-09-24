# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Union
import time
# polyrl-dev
import uuid
import json

import torch
import torch.distributed as dist
# polyrl-dev
import torch.multiprocessing as mp
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
# polyrl-dev
from torch.distributed.tensor import DTensor

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.device import get_device_id, get_torch_device, set_expandable_segments
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.model import convert_weight_keys
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from verl.utils.torch_functional import check_device_is_available
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

# polyrl-dev
from verl.workers.rollout.weight_transfer import (MooncakeTransferEngineConfig,
                                                  TransferAgentConfig,
                                                  start_transfer_agent)
from .base import BaseShardingManager

# polyrl-dev
import requests

# from vllm.distributed import parallel_state as sglang_ps
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def wait_for_rollout_manager_ready(endpoint: str, max_retries: int = 20, initial_delay: float = 1.0, max_delay: float = 30.0) -> bool:
    """
    Wait for rollout manager to become ready with exponential backoff.
    
    Args:
        endpoint: The rollout manager endpoint URL
        max_retries: Maximum number of retry attempts (default: 20)
        initial_delay: Initial delay between retries in seconds (default: 1.0)
        max_delay: Maximum delay between retries in seconds (default: 30.0)
    
    Returns:
        True if rollout manager is ready, False if max retries exceeded
    """
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            # Try to reach the health endpoint or a simple GET request
            health_url = f"{endpoint.rstrip('/')}/health"
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                logger.info(f"\x1b[31;20m[FSDPSGLangShardingManager] Rollout manager is ready at {endpoint}\x1b[31;20m")
                return True
        except Exception as e:
            logger.info(f"[FSDPSGLangShardingManager] Attempt {attempt + 1}/{max_retries}: Rollout manager not ready at {endpoint}, error: {e}")
        
        if attempt < max_retries - 1:  # Don't sleep on the last attempt
            time.sleep(delay)
            delay = min(delay * 1.5, max_delay)  # Exponential backoff with cap
    
    logger.error(f"[FSDPSGLangShardingManager] Failed to connect to rollout manager at {endpoint} after {max_retries} attempts")
    return False


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


@dataclass
class RolloutManagerConfig:
    port: int  # Rollout manager port
    endpoint: str  # Rollout manager HTTP endpoint


class FSDPSGLangShardingManager(BaseShardingManager):
    @check_device_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: Engine,
        model_config,
        rollout_config,
        # polyrl-dev
        weight_sender_config = None,
        # polyrl-dev
        rollout_manager_config = None,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        multi_stage_wake_up: bool = False,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.rollout_config = rollout_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param
        self.multi_stage_wake_up = multi_stage_wake_up

        # polyrl-dev
        if weight_sender_config is not None:
            self.weight_sender_config = self._build_sender_config(weight_sender_config)
        else:
            self.weight_sender_config = None
        self.weight_sender_agent = None
        self.weight_sender_agent_queues = None
        self.weight_sender_agent_buffer = None
        self.rollout_manager_config = rollout_manager_config

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig()
            )
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = get_torch_device().get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        # polyrl-dev
        if self.rollout_manager_config is not None:
            self.wait_for_rollout_manager_ready()

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        self.timing = {}
        with simple_timer("reshard", self.timing):
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.wake_up())
                
    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    def update_weight_remote(self):
        self.timing = {}
        with simple_timer("reshard_remote", self.timing):
            get_torch_device().empty_cache()
            log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
            if self.offload_param:
                load_fsdp_model_to_gpu(self.module)
            params = self.module.state_dict()
            log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            params = {k: v.to(device, non_blocking=True) if fsdp_version(self.module) == 2 else v for k, v in params.items()}
            params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))
            # Copy, not share memory
            # polyrl-dev
            if self.weight_sender_config is not None:
                self.update_weights_with_agent(params)
            else:
                raise ValueError("weight sender config must be set to update with agent!")
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)

            del params
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            get_torch_device().empty_cache()
            log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

            # important: need to manually set the random states of each tp to be identical.
            if self.device_mesh is not None:
                self.torch_random_states = get_torch_device().get_rng_state()
                get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.sleep())

    async def update_weights(self, params):
        named_tensors = [(k, v) for k, v in params.items()]
        update_weights_bucket_bytes = int(self.rollout_config.update_weights_bucket_megabytes) << 20
        for params_batch in get_named_tensor_buckets(named_tensors, update_weights_bucket_bytes):
            # polyrl-dev
            # sgl_update_weights will aggregate tensors and wrap into a request format
            await sgl_update_weights(
                engine=self.inference_engine,
                params_batch=params_batch,
                device_mesh_key="infer_tp",
                device_mesh=self.device_mesh,
            )

        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            await self.inference_engine.flush_cache()

    async def release_memory(self):
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            if self.multi_stage_wake_up:
                await self.inference_engine.release_memory_occupation(tags=["kv_cache", "weights"])
            else:
                await self.inference_engine.release_memory_occupation()
            log_gpu_memory_usage("After release memory occupation in sharding manager", logger=logger)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    async def wake_up(self):
        get_torch_device().empty_cache()

        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)
        params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
        device = get_device_id()  # used when fsdp2 set cpu_offload_policy
        params = {
            k: v.to(device, non_blocking=True) if fsdp_version(self.module) == 2 else v for k, v in params.items()
        }

        # convert weight keys to match the model config
        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)

        log_gpu_memory_usage("After offload_param in sharding manager memory", logger=logger)

        # sglang need to set _set_allocator_settings to False
        logger.debug("fsdp sglang sharding_manager _set_allocator_settings to False")
        set_expandable_segments(False)

        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.rollout_config.free_cache_engine:
            if self.multi_stage_wake_up:
                await self.inference_engine.resume_memory_occupation(tags=["weights"])
                log_gpu_memory_usage("Before resume SGLang weights in sharding manager", logger=logger)
            else:
                await self.inference_engine.resume_memory_occupation()
                log_gpu_memory_usage("Before resume SGLang weights + kv_cache in sharding manager", logger=logger)

        # Copy, not share memory
        await self.update_weights(params)
        log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)

        del params
        get_torch_device().empty_cache()
        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        if (
            self.multi_stage_wake_up
            and self.rollout_config.free_cache_engine
            and self.device_mesh["infer_tp"].get_local_rank() == 0
        ):
            await self.inference_engine.resume_memory_occupation(tags=["kv_cache"])
            log_gpu_memory_usage("After resume SGLang kv_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    async def sleep(self):
        if self.rollout_config.free_cache_engine:
            log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
            await self.release_memory()
            log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        self.module.train()

        # add empty cache after each compute
        get_torch_device().empty_cache()

        # always set _set_allocator_settings to True when using sglang
        # it is required by fsdp2 to avoid oom
        logger.debug("fsdp sglang sharding_manager _set_allocator_settings to True")
        set_expandable_segments(True)

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = get_torch_device().get_rng_state()
            get_torch_device().set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        group = self.device_mesh["infer_tp"].get_group()

        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]

    # polyrl-dev
    def update_weights_with_agent(self, params):
        local_rank = int(os.environ.get('RAY_LOCAL_RANK', 0))
        global_rank = dist.get_rank()
        
        # Every local rank 0 creates and manages its own weight sender agent
        if self.weight_sender_agent is None:
            meta_size, tensors_meta = self._get_meta_tensors_from_state_dict(params)

            if local_rank == 0:
                rollout_manager_endpoint = self.rollout_manager_config.endpoint if self.rollout_manager_config else None
                self._start_transfer_agent(meta_size, tensors_meta, rollout_manager_endpoint)
            else:
                self.weight_sender_agent = False
        
        # Only global rank 0 calls update_weight_version
        if global_rank == 0:
            self._update_weight_version()
        
        # Barrier to ensure all nodes are ready before weight copy
        if self.device_mesh is not None:
            dist.barrier()

        # All ranks copy weights to buffer, but only local rank 0s send, most time consuming step
        tensors_meta = self._copy_weights_to_buffer(params)

        if local_rank == 0:
            self.weight_sender_agent_queues[0].put("update_weights")
            status = self.weight_sender_agent_queues[1].get()
            if status != "completed":
                raise RuntimeError(f"Weight update failed: {status}")
        
        # Barrier for FSDP synchronization only
        if self.device_mesh is not None:
            dist.barrier()

    # polyrl-dev
    def _update_weight_version(self):
        if self.rollout_manager_config is None:
            logger.warning("[Weight Transfer] Rollout manager config not set, skipping weight version update")
            return
        
        rollout_mgr_url = self.rollout_manager_config.endpoint
        try:
            response = requests.post(f"{rollout_mgr_url.rstrip('/')}/update_weight_version", timeout=60)
            response.raise_for_status()
            result = response.json()
            if result.get("success", False):
                new_version = result.get("new_weight_version", 0)
                logger.info(f"[Weight Transfer] Successfully updated weight version to {new_version}")
            else:
                raise RuntimeError(f"Rollout manager weight version update failed: {response.text}")
        except Exception as e:
            logger.error(f"Error during weight version update call to rollout manager: {e}")
            raise

    def _build_sender_config(self, config):
        # Extract configuration values from self.config
        assert 'RAY_LOCAL_RANK' in os.environ, "RAY_LOCAL_RANK must be set"
        assert 'WEIGHT_SENDER_IP' in os.environ, "WEIGHT_SENDER_IP must be set"
        
        weight_sender_ips = os.environ.get('WEIGHT_SENDER_IP').split(',')
        num_mooncake_groups = config.get('num_mooncake_groups', 1)
        num_mooncake_engines_per_group = config.get('num_mooncake_engines_per_group', 1)
        
        if len(weight_sender_ips) != num_mooncake_groups:
            raise ValueError(f"Number of weight sender IPs ({len(weight_sender_ips)}) does not match num_mooncake_groups ({num_mooncake_groups})")
        rpyc_bind_port = config.get('rpyc_bind_base_port', 18861)
        transfer_protocol = config.get('transfer_protocol', 'tcp')
        transfer_device_name = config.get('transfer_device_name', '')
        base_handshake_port = config.get('mooncake_handshake_port', 19000)
        
        # Create Mooncake configurations for each group
        mooncake_configs = []
        for group_idx, ip in enumerate(weight_sender_ips):
            group_handshake_port = base_handshake_port + group_idx * 1000
            mooncake_config = MooncakeTransferEngineConfig(
                local_hostname=ip,
                protocol=transfer_protocol,
                device_name=transfer_device_name,
                handshake_port=group_handshake_port,
            )
            mooncake_configs.append(mooncake_config)
        global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size() 
        # Create and return the sender configuration
        sender_config = TransferAgentConfig(
            trainer_global_rank=global_rank,
            trainer_world_size=world_size,
            mooncake_config=mooncake_configs,
            num_mooncake_engines_per_group=num_mooncake_engines_per_group,
            rpyc_bind_port=rpyc_bind_port,
        )
        
        return sender_config

    # polyrl-dev
    def _get_meta_tensors_from_state_dict(self, state_dict):
        meta_size = []
        tensors_meta = []
        for name, param in state_dict.items():
            if isinstance(param, DTensor):
                param = param.full_tensor()
            assert torch.is_tensor(param)
            # meta = torch.empty_like(param, device='meta')
            meta_size.append((name, param.numel() * param.element_size()))

            dtype = str(param.dtype).split('.')[-1]
            shape = list(param.shape)
            tensors_meta.append((name, (shape, dtype)))
        return meta_size, tensors_meta

    # polyrl-dev
    # NOTE(yongji): Lazy initialization of weight transfer on the first weight transfer
    def _start_transfer_agent(self, meta_tensors, tensors_meta, rollout_manager_endpoint):
        # Create TCP topology JSON if using TCP protocol
        if hasattr(self.weight_sender_config.mooncake_config, 'protocol') and \
            self.weight_sender_config.mooncake_config.protocol == 'tcp':
            create_tcp_topology_json()
        
        os.environ["MC_FORCE_TCP"] = "1"
        os.environ["MC_LEGACY_RPC_PORT_BINDING"] = "1"
        mp.set_start_method('spawn', force=True)
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        input_queue.put(meta_tensors)
        input_queue.put(tensors_meta)
        self.weight_sender_agent = start_transfer_agent(
            self.weight_sender_config, 
            input_queue, 
            output_queue,
            rollout_manager_endpoint,
        )
        self.weight_sender_agent_queues = (input_queue, output_queue)
        
        result = output_queue.get(timeout=60)
        if isinstance(result, tuple):
            shm_path, buffer_length = result
            assert isinstance(shm_path, str), "Should receive shared memory path"
            assert isinstance(buffer_length, int), "Should receive buffer length"
            
            from verl.workers.rollout.weight_transfer.agent import create_tensor_from_shared_memory
            self.weight_sender_agent_buffer = create_tensor_from_shared_memory(shm_path, buffer_length)
        else:
            buffer = result
            assert torch.is_tensor(buffer)
            assert buffer.is_cpu
            self.weight_sender_agent_buffer = buffer

    # polyrl-dev
    def _copy_weights_to_buffer(self, state_dict):
        offset = 0
        tensors_meta = []
        for name, param in state_dict.items():
            if isinstance(param, DTensor):
                param = param.full_tensor()
            numel = param.numel()
            size_in_bytes = numel * param.element_size()
            
            if self.weight_sender_agent:
                param_data_cpu = param.data.contiguous() # .cpu()
                param_u8 = param_data_cpu.view(-1).view(torch.uint8)
                buffer_slice = self.weight_sender_agent_buffer[offset : offset + size_in_bytes]
                buffer_slice.copy_(param_u8, non_blocking=True)
                offset += size_in_bytes

            dtype = str(param.dtype).split('.')[-1]
            shape = list(param.shape)
            tensors_meta.append((name, (shape, dtype)))
        torch.cuda.synchronize()

        return tensors_meta

    def wait_for_rollout_manager_ready(self):
        if self.rollout_manager_config is None:
            return
        rollout_mgr_endpoint = self.rollout_manager_config.endpoint
        if not wait_for_rollout_manager_ready(rollout_mgr_endpoint):
            raise RuntimeError("Rollout manager is not ready")


def create_tcp_topology_json():
    # Generate base filename
    base_filename = "mc_topo.json"
    target_path = f"/tmp/{base_filename}"
    
    # Try to remove existing file, if it exists and can't be removed, add random suffix
    if os.path.exists(target_path):
        try:
            os.remove(target_path)
            logger.info(f"Removed existing file: {target_path}")
        except OSError:
            # Can't remove, add uuid 4-digit suffix
            uuid_suffix = str(uuid.uuid4()).replace('-', '')[:4]
            base_filename = f"mc_topo_{uuid_suffix}.json"
            target_path = f"/tmp/{base_filename}"
            logger.info(f"Could not remove existing file, using new name: {target_path}")
    
    # Create empty JSON file
    try:
        with open(target_path, 'w') as f:
            json.dump({}, f)
        logger.info(f"Created empty topology JSON file: {target_path}")
        
        # Set environment variable
        os.environ["MC_CUSTOM_TOPO_JSON"] = target_path
        logger.info(f"Set MC_CUSTOM_TOPO_JSON environment variable to: {target_path}")
        
    except Exception as e:
        logger.error(f"Failed to create topology JSON file: {e}")
        raise
