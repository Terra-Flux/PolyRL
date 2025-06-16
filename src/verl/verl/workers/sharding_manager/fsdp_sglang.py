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
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.utils import MultiprocessingSerializer
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import (FullStateDictConfig,
                                        ShardedStateDictConfig, StateDictType)
from torch.distributed.fsdp.fully_sharded_data_parallel import \
    FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.fsdp_utils import (fsdp_version, load_fsdp_model_to_gpu,
                                   offload_fsdp_model_to_cpu)
from verl.utils.torch_functional import check_cuda_is_available
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
                logger.info(f"[FSDPSGLangShardingManager] Rollout manager is ready at {endpoint}")
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
    @check_cuda_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: Engine,
        model_config,
        # polyrl-dev
        weight_sender_config = None,
        # polyrl-dev
        rollout_manager_config = None,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param

        # polyrl-dev
        if weight_sender_config is not None:
            self.weight_sender_config = self._build_sender_config(weight_sender_config)
        else:
            self.weight_sender_config = None
        self.weight_sender_agent = None
        self.weight_sender_agent_queues = None
        self.weight_sender_agent_buffer = None
        self.rollout_manager_config = rollout_manager_config
        
        # Store tensors_meta to avoid re-computation in subsequent updates
        self.cached_tensors_meta = None

        # Full params
        self.full_params = full_params
        if full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

        # polyrl-dev
        if self.rollout_manager_config is not None:
            self.wait_for_rollout_manager_ready()

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        torch.cuda.empty_cache()
        log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
        if self.offload_param:
            load_fsdp_model_to_gpu(self.module)
        params = self.module.state_dict()
        log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
        device = torch.cuda.current_device()  # used when fsdp2 set cpu_offload_policy
        params = {k: v.to(device, non_blocking=True) if fsdp_version(self.module) == 2 else v for k, v in params.items()}
        # Copy, not share memory
        # polyrl-dev
        if self.weight_sender_config is not None:
            self.update_weights_with_agent(params)
        else:
            self.update_weights(params)
        log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)

        del params
        if self.offload_param:
            offload_fsdp_model_to_cpu(self.module)
        torch.cuda.empty_cache()
        log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        self.release_memory()
        log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def update_weights(self, params):
        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            self.inference_engine.resume_memory_occupation()

        # Most naive implementation, can optimize a lot if it is bottleneck from sglang Engine weight update
        named_tensors = [(k, v) for k, v in params.items()]
        load_format = None
        for tensor_index, (name, tensor) in enumerate(named_tensors):
            serialized_tensor = MultiprocessingSerializer.serialize(_preprocess_tensor_for_update_weights(tensor))

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                gathered_serialized_tensors = [None for _ in range(self.device_mesh["infer_tp"].mesh.size()[0])]
            else:
                gathered_serialized_tensors = None
            dist.gather_object(
                obj=serialized_tensor,
                object_gather_list=gathered_serialized_tensors,
                dst=self.device_mesh["infer_tp"].mesh.tolist()[0],
                group=self.device_mesh["infer_tp"].get_group(),
            )

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                self.inference_engine.update_weights_from_tensor(
                    named_tensors=[
                        (
                            name,
                            LocalSerializedTensor(values=gathered_serialized_tensors),
                        )
                    ],
                    load_format=load_format,
                    flush_cache=tensor_index == len(named_tensors) - 1,
                )

    def release_memory(self):
        # polyrl-dev
        if self.device_mesh["infer_tp"].get_local_rank() == 0 and self.inference_engine is not None:
            self.inference_engine.release_memory_occupation()

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
        
        # Each local rank 0 gets its own receivers from rollout manager
        current_receivers = []
        if local_rank == 0:
            weight_sender_endpoint = f"{os.environ.get('RAY_LOCAL_IP')}:{self.weight_sender_config.rpyc_bind_port}"
            
            if self.rollout_manager_config is not None:
                try:
                    # Get current instances for this specific weight sender
                    rollout_mgr_url = self.rollout_manager_config.endpoint
                    payload = {"weight_sender_endpoint": weight_sender_endpoint}
                    # polyrl-dev
                    # NOTE(liuxs): change timeout based on your hand speed
                    response = requests.post(f"{rollout_mgr_url.rstrip('/')}/prepare_weight_update", json=payload, timeout=600)
                    response.raise_for_status()
                    prepare_data = response.json()
                    current_instances = prepare_data.get("current_instances", [])
                    
                    # Extract receiver info from instances assigned to this weight sender
                    for instance in current_instances:
                        endpoint = instance["endpoint"].replace("http://", "").replace("https://", "")
                        current_receivers.append({
                            "instance_id": instance["id"],
                            "endpoint": endpoint
                        })
                        
                    logger.info(f"[Weight sender] Current receivers for sender {weight_sender_endpoint}: {current_receivers}")
                except Exception as e:
                    logger.error(f"Failed to get current instances from rollout manager: {e}")
                    raise
        
        # Every local rank 0 creates and manages its own weight sender agent
        if self.weight_sender_agent is None:
            meta_tensors, tensors_meta = self._get_meta_tensors_from_state_dict(params)

            # Cache tensors_meta on global rank 0 to avoid re-computation
            if global_rank == 0:
                self.cached_tensors_meta = tensors_meta

            if local_rank == 0:
                self._start_transfer_agent(meta_tensors)
            else:
                self.weight_sender_agent = False
        
        # TODO(yongji): Optimize to only notify newly joined SGLang instances instead of broadcasting to all instances every round
        # Global rank 0 always sends bootstrap notification for potential new receivers
        if global_rank == 0:
            self._notify_weight_update(self.cached_tensors_meta, bootstrap=True)
        
        # Wait for receivers on each weight update
        if local_rank == 0 and current_receivers:
            # Extract receiver IDs to wait for
            receiver_ids = []
            for receiver in current_receivers:
                # Use endpoint as receiver ID for now
                receiver_ids.append(receiver["endpoint"])
            
            # Send wait request - sender agent will handle checking if receivers are already registered
            self.weight_sender_agent_queues[0].put("wait_for_receivers")
            self.weight_sender_agent_queues[0].put(receiver_ids)
            status = self.weight_sender_agent_queues[1].get()
            if status != "completed":
                raise RuntimeError(f"Waiting for receivers failed: {status}")
            logger.info(f"[Weight Transfer] Local rank {local_rank} receivers ready")
        
        # Barrier to ensure all nodes have completed prepare_weight_update before proceeding
        if self.device_mesh is not None:
            logger.info(f"[Weight Transfer] Waiting for all nodes to complete prepare_weight_update...")
            dist.barrier()
            logger.info(f"[Weight Transfer] All nodes have completed prepare_weight_update")

        # All ranks copy weights to buffer, but only local rank 0s send
        tensors_meta = self._copy_weights_to_buffer(params)

        if local_rank == 0:
            logger.info(f"[Weight Transfer] Local rank {local_rank} sending weights...")
            self.weight_sender_agent_queues[0].put("send_weights")
            status = self.weight_sender_agent_queues[1].get()
            if status != "completed":
                raise RuntimeError(f"Weight send failed: {status}")
            logger.info(f"[Weight Transfer] Local rank {local_rank} weights sent successfully")
        
        # Only global rank 0 notifies weight update
        if global_rank == 0:
            self._notify_weight_update(tensors_meta, bootstrap=False)
        
        # Barrier to ensure weight update is completed before rollout on all GPUs
        dist.barrier()

    # polyrl-dev
    def _notify_weight_update(self, tensors_meta, bootstrap=False):
        if self.inference_engine is None: # Rollout-manager mode
            rollout_mgr_url = self.rollout_manager_config.endpoint
            payload = {
                "tensors_meta": tensors_meta,
                "load_format": None, # Or pass as needed
                "flush_cache": True, # Or pass as needed
                "bootstrap": bootstrap
            }
            try:
                # polyrl-dev
                # NOTE(liuxs): change timeout based on your hand speed
                response = requests.post(f"{rollout_mgr_url.rstrip('/')}/update_weights_from_agent", json=payload, timeout=600)
                response.raise_for_status()
                if not response.json().get("success", False):
                    raise RuntimeError(f"Rollout manager weight update failed: {response.text}")
                logger.info("Rollout manager successfully updated weights.")
            except Exception as e:
                logger.error(f"Error during weight update call to rollout manager: {e}")
                raise

    def _build_sender_config(self, config):
        # Extract configuration values from self.config
        assert 'RAY_LOCAL_RANK' in os.environ, "RAY_LOCAL_RANK must be set"
        assert 'RAY_ROOT_IP' in os.environ, "RAY_ROOT_IP must be set"
        assert 'RAY_LOCAL_IP' in os.environ, "RAY_LOCAL_IP must be set"

        local_hostname = os.environ.get('RAY_LOCAL_IP')
        rpyc_bind_port = config.get('rpyc_bind_base_port', 18861)
        transfer_protocol = config.get('transfer_protocol', 'tcp')
        transfer_device_name = config.get('transfer_device_name', '')
        mooncake_config_path = config.get('mooncake_config_path', '')

        # Create Mooncake configuration
        if mooncake_config_path:
            mooncake_config = mooncake_config_path
        else:
            # Use P2P handshake instead of etcd
            handshake_port = config.get('mooncake_handshake_port', 19000)
            mooncake_config = MooncakeTransferEngineConfig(
                local_hostname=local_hostname,
                protocol=transfer_protocol,
                device_name=transfer_device_name,
                handshake_port=handshake_port,
            )
        global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size() 
        # Create and return the sender configuration
        sender_config = TransferAgentConfig(
            trainer_global_rank=global_rank,
            trainer_world_size=world_size,
            mooncake_config=mooncake_config,
            rpyc_bind_port=rpyc_bind_port,
            mooncake_handshake_port=handshake_port if 'handshake_port' in locals() else None
        )
        
        return sender_config

    # polyrl-dev
    def _get_meta_tensors_from_state_dict(self, state_dict):
        meta_tensors = []
        tensors_meta = []
        for name, param in state_dict.items():
            if isinstance(param, DTensor):
                param = param.full_tensor()
            assert torch.is_tensor(param)
            meta = torch.empty_like(param, device='meta')
            meta_tensors.append((name, meta))

            dtype = str(param.dtype).split('.')[-1]
            shape = list(param.shape)
            tensors_meta.append((name, (shape, dtype)))
        return meta_tensors, tensors_meta

    # polyrl-dev
    # NOTE(yongji): Lazy initialization of weight transfer on the first weight transfer
    def _start_transfer_agent(self, meta_tensors):
        # Create TCP topology JSON if using TCP protocol
        if hasattr(self.weight_sender_config.mooncake_config, 'protocol') and \
            self.weight_sender_config.mooncake_config.protocol == 'tcp':
            create_tcp_topology_json()
        
        mp.set_start_method('spawn', force=True)
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        input_queue.put(meta_tensors)
        self.weight_sender_agent = start_transfer_agent(
            self.weight_sender_config, 
            input_queue, 
            output_queue,
        )
        self.weight_sender_agent_queues = (input_queue, output_queue)
        buffer = output_queue.get(timeout=60)
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
                param_data_cpu = param.data.contiguous().cpu()
                param_u8 = param_data_cpu.view(-1).view(torch.uint8)
                buffer_slice = self.weight_sender_agent_buffer[offset : offset + size_in_bytes]
                buffer_slice.copy_(param_u8)
                offset += size_in_bytes

            dtype = str(param.dtype).split('.')[-1]
            shape = list(param.shape)
            tensors_meta.append((name, (shape, dtype)))

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
