# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""A tensor parallel worker."""

import logging
import threading
from typing import Optional, Tuple, Union
# polyrl-dev
import os
import json
import uuid

import torch
# polyrl-dev
import torch.multiprocessing as mp

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import (
    GetWeightsByNameReqInput,
    InitWeightsUpdateGroupReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromTensorReqInput,
    # polyrl-dev
    UpdateWeightsFromAgentReqInput,
)
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, global_server_args_dict
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj, set_random_seed

# polyrl-dev
from sglang.srt.weight_transfer.utils import TransferAgentConfig, MooncakeTransferEngineConfig
from sglang.srt.weight_transfer.agent import start_transfer_agent

logger = logging.getLogger(__name__)


class TpModelWorker:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        pp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[TokenToKVPoolAllocator] = None,
    ):
        # Parse args
        self.server_args = server_args  # polyrl-dev: Save server_args for later use
        self.tp_size = server_args.tp_size
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank

        # Init model and tokenizer
        self.model_config = ModelConfig.from_server_args(
            server_args,
            model_path=(
                server_args.model_path
                if not is_draft_worker
                else server_args.speculative_draft_model_path
            ),
            is_draft_model=is_draft_worker,
        )

        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            pp_rank=pp_rank,
            pp_size=server_args.pp_size,
            nccl_port=nccl_port,
            server_args=server_args,
            is_draft_worker=is_draft_worker,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                )
        self.device = self.model_runner.device

        # Init nccl groups
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
                // (server_args.dp_size if server_args.enable_dp_attention else 1)
            ),
            self.model_runner.req_to_token_pool.size,
        )
        assert self.max_running_requests > 0, "max_running_request is zero"
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_size * self.pp_rank + tp_rank,
            self.world_group.cpu_group,
            src=self.world_group.ranks[0],
        )[0]
        set_random_seed(self.random_seed)

        # A reference make this class has the same member as TpModelWorkerClient
        self.worker = self

        # polyrl-dev
        if server_args.enable_weight_transfer_agent:
            self.weight_receiver_config = self._build_weight_receiver_config(server_args)
            self.weight_receiver_agent = None
            self.weight_receiver_agent_queues = None
            self.weight_receiver_agent_buffer = None

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            global_server_args_dict,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_tp_group(self):
        return self.model_runner.tp_group

    def get_attention_tp_group(self):
        return self.model_runner.attention_tp_group

    def get_attention_tp_cpu_group(self):
        return getattr(self.model_runner.attention_tp_group, "cpu_group", None)

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        launch_done: Optional[threading.Event] = None,
        skip_sample: bool = False,
    ) -> Tuple[
        Union[LogitsProcessorOutput, torch.Tensor], Optional[torch.Tensor], bool
    ]:
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)

        pp_proxy_tensors = None
        if not self.pp_group.is_first_rank:
            pp_proxy_tensors = PPProxyTensors(
                self.pp_group.recv_tensor_dict(
                    all_gather_group=self.get_attention_tp_group()
                )
            )

        if self.pp_group.is_last_rank:
            logits_output, can_run_cuda_graph = self.model_runner.forward(
                forward_batch, pp_proxy_tensors=pp_proxy_tensors
            )
            if launch_done is not None:
                launch_done.set()

            if skip_sample:
                next_token_ids = None
            else:
                next_token_ids = self.model_runner.sample(
                    logits_output, model_worker_batch
                )

            return logits_output, next_token_ids, can_run_cuda_graph
        else:
            pp_proxy_tensors, can_run_cuda_graph = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            return pp_proxy_tensors.tensors, None, can_run_cuda_graph

    def forward_batch_embedding(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output, _ = self.model_runner.forward(forward_batch)
        embeddings = logits_output.embeddings
        return embeddings

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.model_runner.update_weights_from_disk(
            recv_req.model_path, recv_req.load_format
        )
        return success, message

    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.model_runner.init_weights_update_group(
            recv_req.master_address,
            recv_req.master_port,
            recv_req.rank_offset,
            recv_req.world_size,
            recv_req.group_name,
            recv_req.backend,
        )
        return success, message

    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = self.model_runner.update_weights_from_distributed(
            recv_req.name, recv_req.dtype, recv_req.shape
        )
        return success, message

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=MultiprocessingSerializer.deserialize(
                recv_req.serialized_named_tensors[self.tp_rank]
            ),
            load_format=recv_req.load_format,
        )
        return success, message
    
    # polyrl-dev
    def update_weights_from_agent(self, recv_req: UpdateWeightsFromAgentReqInput):
        tensors_meta = recv_req.tensors_meta
        if self.weight_receiver_agent is None and self.tp_rank == 0:
            assert recv_req.bootstrap
            meta_tensors = []
            for name, (shape, dtype) in tensors_meta:
                dtype = getattr(torch, dtype)
                meta = torch.empty(shape, dtype=dtype, device='meta')
                meta_tensors.append((name, meta))
            self._start_weight_receiver_agent(meta_tensors, self.server_args)
            return True, "Success"
        elif recv_req.bootstrap:
            return True, "Success"

        assert recv_req.bootstrap is False
        if self.tp_rank == 0:
            self.weight_receiver_agent_queues[0].put("receive_weights")
            status = self.weight_receiver_agent_queues[1].get()
            if status != "completed":
                raise RuntimeError(f"Weight receive failed or unexpected status: {status}")
        if self.tp_rank == 0:
            named_tensors = self._construct_received_weights(tensors_meta)
            for name, tensor in named_tensors:
                tensor = tensor.to(self.device)
                torch.distributed.broadcast(tensor, src=0, group=self.world_group.device_group)
                success, message = self.model_runner.update_weights_from_tensor(
                    named_tensors=[(name, tensor)],
                    load_format=recv_req.load_format,
                    unwrap_tensor=False,
                )
                assert success
        else:
            for name, (shape, dtype) in tensors_meta:
                dtype = getattr(torch, dtype)
                tensor = torch.empty(shape, dtype=dtype, device=self.device)
                torch.distributed.broadcast(tensor, src=0, group=self.world_group.device_group)
                success, message = self.model_runner.update_weights_from_tensor(
                    named_tensors=[(name, tensor)],
                    load_format=recv_req.load_format,
                    unwrap_tensor=False,
                )
                assert success
        return success, message

    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.model_runner.get_weights_by_name(
            recv_req.name, recv_req.truncate_size
        )
        return parameter

    # polyrl-dev
    def _build_weight_receiver_config(self, server_args: ServerArgs):
        host, port = server_args.weight_sender_rpyc_endpoint.split(":")
        sender_rpyc_endpoints = [(host, int(port))]

        if os.getenv("MOONCAKE_CONFIG_PATH") is None:
            mooncake_config = MooncakeTransferEngineConfig(
                local_hostname=server_args.host,
                protocol=server_args.mooncake_transfer_protocol,
                device_name=server_args.mooncake_transfer_device_name,
                handshake_port=server_args.mooncake_handshake_port,
            )
        else:
            mooncake_config = None
        config = TransferAgentConfig(
            sender_rpyc_endpoints=sender_rpyc_endpoints,
            mooncake_config=mooncake_config,
            sglang_http_host=server_args.host,
            sglang_http_port=server_args.port,
            zmq_bind_host=server_args.weight_receiver_zmq_bind_host,
            mooncake_handshake_port=server_args.mooncake_handshake_port,
        )
        return config
    
    # polyrl-dev
    def _start_weight_receiver_agent(self, meta_tensors, server_args):
        assert self.weight_receiver_agent is None
        
        # Create TCP topology JSON if using TCP protocol
        if hasattr(self.weight_receiver_config.mooncake_config, 'protocol') and \
            self.weight_receiver_config.mooncake_config.protocol == 'tcp':
            create_tcp_topology_json()
        
        mp.set_start_method('spawn', force=True)
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        input_queue.put(meta_tensors)
        self.weight_receiver_agent = start_transfer_agent(
            self.weight_receiver_config, 
            input_queue, 
            output_queue,
            server_args
        )
        self.weight_receiver_agent_queues = (input_queue, output_queue)
        buffer = output_queue.get(timeout=60)
        assert torch.is_tensor(buffer)
        assert buffer.is_cpu
        self.weight_receiver_agent_buffer = buffer

    # polyrl-dev
    def _construct_received_weights(self, tensors_meta):
        offset = 0
        named_tensors = []
        for name, (shape, dtype) in tensors_meta:
            size = 1
            for dim in shape:
                size *= dim
            dtype = getattr(torch, dtype)
            elem_size = torch.finfo(dtype).bits // 8
            size_in_bytes = size * elem_size
            buffer_slice = self.weight_receiver_agent_buffer[offset : offset + size_in_bytes]
            tensor = buffer_slice.view(dtype).view(*shape)
            named_tensors.append((name, tensor))
            offset += size_in_bytes
        return named_tensors


# polyrl-dev
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
