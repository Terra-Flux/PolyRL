# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import logging
import time
import uuid
from collections import deque
from typing import Iterator, List, Optional, Tuple

import aiohttp
import numpy as np
import torch
from tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence
import os
import requests

from sglang.srt.utils import (
    get_ip,
    get_open_port,
)

from verl import DataProto
from verl.utils.device import get_torch_device
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from .utils import broadcast_pyobj
from .sglang_rollout import _pre_process_inputs, _post_process_outputs, SGLangRollout
from verl.utils.net_utils import is_ipv6

from .stream_batch_iter import StreamingBatchIterator
from .sglang_http_async_engine import HttpServerAsyncEngineAdapter

from ray import get_actor, get
import torch.distributed as dist

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# TODO(liuxs): because generate_sequence is an dispatch call, we cannot integrate the local/remote split into one function
# However, we may consider execute all-gather and let rank0 submit the requests

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
                logger.info(f"\x1b[31;20m[SGLangRollout] Rollout manager is ready at {endpoint}\x1b[31;20m")
                return True
        except Exception as e:
            logger.info(f"[SGLangRollout] Attempt {attempt + 1}/{max_retries}: Rollout manager not ready at {endpoint}, error: {e}")
        
        if attempt < max_retries - 1:  # Don't sleep on the last attempt
            time.sleep(delay)
            delay = min(delay * 1.5, max_delay)  # Exponential backoff with cap
    
    logger.error(f"[SGLangRollout] Failed to connect to rollout manager at {endpoint} after {max_retries} attempts")
    return False

class SGLangRolloutRemote(SGLangRollout):
    """
    Iterator wrapper for SGLang rollout in disaggregated mode.
    Submits requests to rollout manager and yields ibatches as responses arrive.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # polyrl-dev
        # minimum unit the rollout iter returns, dividable by both sample_n and update_actor_ibatch.
        self.sample_n = self.config.get("n", 1)
        assert self.config.min_stream_batch_size % self.sample_n == 0, f"min_stream_batch_size {self.config.min_stream_batch_size} must be dividable by {self.sample_n=}"
        self.min_stream_prompts = self.config.min_stream_batch_size // self.sample_n
        logger.info(f"[PolyRL] Default rollout iterator batch size is {self.min_stream_prompts} prompts")
        
    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        # initialize the inference engine
        nnodes = -(-self._tp_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group("tp"),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None

        load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        assert self.config.name == 'sglang-disaggregated', f"Only sglang-disaggregated rollout should init SGLangRolloutRemote, got {self.config.name}"
        self._engine = None 
        # Wait for rollout manager to be ready before proceeding
        logger.info("[SGLangRollout] Waiting for rollout manager to be ready")
        if self.config.rollout_manager.endpoint:
            rollout_mgr_endpoint = self.config.rollout_manager.endpoint
            if not wait_for_rollout_manager_ready(rollout_mgr_endpoint):
                raise RuntimeError(f"[SGLangRollout] Rollout manager at {rollout_mgr_endpoint} is not ready after maximum retries")
        
        # polyrl-dev
        # init SGLang engine for buffer zone
        if first_rank_in_node:
            rank = dist.get_rank()
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            # self._engine = AsyncEngine(
            self._engine = HttpServerAsyncEngineAdapter(
                rollout_mgr_endpoint=rollout_mgr_endpoint, # additional argument to specify rollout manager address
                model_path=actor_module,
                dtype=self.config.dtype,
                mem_fraction_static=self.config.gpu_memory_utilization,
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes,
                trust_remote_code=trust_remote_code,
                # NOTE(linjunrong): add rank to prevent SGLang generate same port inside PortArgs.init_new
                # when random.seed is being set during training
                port=30000 + rank,
                log_level="warning",
                # NOTE(Chenyang): if you want to debug the SGLang engine output
                # please set the following parameters
                # Otherwise, it will make the engine run too slow
                # log_level="INFO",
                # log_requests=True,
                # log_requests_level=2,
                # max_running_requests=1,
                mm_attention_backend="fa3",
            )
        else:
            self._engine = None

        self.sharding_manager = None
        self.is_sleep = True

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # NOTE(liuxs): copy of original colocated rollout
        # sglang_outputs = super().generate_sequences(prompts, **kwargs)
        # self.task_queue.put_nowait(None)
        # return sglang_outputs
        # NOTE(liuxs): validation fall back to original rollout as we don't do remote validation
        is_validate = prompts.meta_info.get("validate", False)
        assert is_validate, "Only call generate sequences when doing validation"
        # TODO(liuxs): local+remote validate generation
        # redirect to sglang rollout  
        sglang_outputs = super().generate_sequences(prompts, **kwargs)
        return sglang_outputs
    
    def _create_generate_remote_payload(self, prompts: DataProto, **kwargs):
        # polyrl-dev
        # unroll samples of same prompt into a list of request batches
        # input ids: (bs, prompt_length), left-padded
        # batch_size = prompts.batch.size(0)
        
        idx_list, image_list, non_tensor_batch, request_sampling_params = self.preprocess_batch(prompts, **kwargs)
        
        # NOTE(liuxs): the new ray_trainer.fit repeat the request before submit, so the original n is always 1
        sample_n = self.sample_n
        print(f"{request_sampling_params=}")
        unroll_sampling_params = request_sampling_params.copy()
        # in new version, verl has already repeat the request, the update is unnecessary actually
        unroll_sampling_params.update({"n" : 1}) # unroll into 1 sample per prompt and submit independent requests
        
        batch_requests = []
        for i, (prompt_idx, image_data) in enumerate(zip(idx_list, image_list)):
            request_payload = {
                "input_ids": [prompt_idx for _ in range(sample_n)], # unroll sample_n
                "sampling_params": unroll_sampling_params,
                "return_logprob": True,
            }
            if image_data:
                request_payload["image_data"] = [image_data for _ in range(sample_n)]
            
            batch_requests.append((i, request_payload))
            
        return batch_requests, non_tensor_batch
            
    def preprocess_batch(self, prompts: DataProto, **kwargs):
        """
        Preprocess the inputs before feeding into SGLang rollout
        """
        # NOTE(liuxs): this part is aligned with SGLangRollout._batch_level_generate_sequences
        # 1. Get meta data required to reconstruct the ibatch results in the postprocess
        # input ids: (bs, prompt_length), left-padded
        idx = prompts.batch["input_ids"]
        batch_size = idx.size(0)
        # Extract non-tensor data
        non_tensor_batch = prompts.non_tensor_batch
        # raw_prompt_ids is now a must-have in ray_trainer
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
                dtype=object,
            )

        if "multi_modal_data" in non_tensor_batch:
            sglang_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
            ):
                sglang_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                        "image_data": (multi_modal_data.get("image", None) if isinstance(multi_modal_data, dict) else None),
                    }
                )
        # polyrl-dev
        # in continuous generation, the image data has been preprocessed
        elif "image_data" in non_tensor_batch:
            raise KeyError(f"Continue generation is migrated to Rollout manager.")
        else:
            sglang_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in sglang_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])


        # Extract token IDs and image data for SGLang Engine
        idx_list = [input_data["prompt_token_ids"] for input_data in sglang_inputs]
        image_list = [input_data.get("image_data", None) for input_data in sglang_inputs]
        
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        
        # Create request-level sampling parameters
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update(
                {
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.0,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "ignore_eos": False,
                    "min_new_tokens": 0,
                    "max_new_tokens": self.config.response_length,
                    "skip_special_tokens": True,
                    "spaces_between_special_tokens": True,
                }
            )
        elif is_validate:
            request_sampling_params.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }
            )

        # Update with any additional kwargs
        request_sampling_params.update(kwargs)
        
        return idx_list, image_list, non_tensor_batch, request_sampling_params
    
    def postprocess_samples(self, out: List[torch.Tensor], 
                            idx: torch.Tensor, attention_mask: torch.Tensor, 
                            position_ids: torch.Tensor, eos_token_id: int,
                            non_tensor_batch: dict) -> DataProto:
        """
        Process samples of responses with its corresponding metadata
        Assume the input_meta_data = global_meta_date[indices]
        
        Args:
            responses: Rollout results (tokens, logprob), padded to the same length (<= response length)
            idx: Tensor of input ids of the prompts in the original batch, left padded to prompt length
            attention_mask: Tensor of attention mask of the prompts in the original batch
            position_ids: Tensor of position ids of the prompts in the original batch
            eos_token_id: EOS token ID for attention mask generation
            non_tensor_batch: Non-tensor batch data
            
        Returns:
            DataProto object containing the processed ibatch
        """
        if not out:
            raise ValueError(f"Postprocessing cannot accept empty response")
        
        device = idx.device
        response = out[0].to(device)  # (batch_size, response_length) in tokens
        rollout_log_probs = None
        if self.config.calculate_log_probs:
            rollout_log_probs = out[1].to(device)  # (batch_size, response_length)
        
        batch_size = response.size(0) # already unrolled on sample_n > 1, no need for * sample_n
        
        # Pad responses to the configured response length if needed
        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_sequence_to_length(rollout_log_probs, self.config.response_length, self.pad_token_id)

        assert response.shape[1] == self.config.response_length, f"Response length {response.shape[1]} after padding is not equal to {self.config.response_length}"
        
        _non_tensor_batch = non_tensor_batch

        # Following the standard SGLang rollout pattern for constructing sequences
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)

        # Create sequence by concatenating prompts and responses
        seq = torch.cat([idx, response], dim=-1)
        
        # Update position IDs following standard pattern
        # NOTE(liuxs): response position_ids = prompt_length + [1,2,3,4...]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        final_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        # Create response attention mask following standard pattern
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        final_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # Create the batch tensor dict following standard SGLang rollout format
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": final_attention_mask,
                "response_mask": response_attention_mask, # we need to return the response mask for the whole ibatch
                "position_ids": final_position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs
            
        logger.debug(f"[SGLangRollout] Rollout batch size is {batch_size}")
        return DataProto(batch=batch, non_tensor_batch=_non_tensor_batch)
    
    def _launch_generate_remote(self, prompts: DataProto, stream_size: int, **kwargs) -> Iterator[DataProto]:
        # polyrl-dev
        # get meta data required to reconstruct the ibatch results in the postprocess
        # input ids: (bs, prompt_length), left-padded
        idx = prompts.batch["input_ids"]
        # attention_mask: (bs, seq_length), left-padded
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to generate attention mask for the
        # response based on EOS token position
        eos_token_id = prompts.meta_info["eos_token_id"]
        
        # 3. Try to get the current running loop, if none exists create a new one
        payload, non_tensor_batch = self._create_generate_remote_payload(prompts, **kwargs)
        # polyrl-dev
        rollout_mgr_url = self.config.rollout_manager.endpoint
        assert rollout_mgr_url is not None, "rollout_manager.endpoint must be set"
        batch_iter = StreamingBatchIterator(url=f"{rollout_mgr_url.rstrip('/')}/batch_generate_requests", payload=payload)
        
        # NOTE(liuxs): this is just a checkpoint that stream has start, otherwise the iterator won't begin until the __next__
        notifier = batch_iter.__next__()
        logger.info(f"Receive notifier {notifier}")
        yield notifier.get("success", False)
        
        # NOTE(liuxs): because we submit the samples of the same prompt in a batch, 
        # here the return of batch iter is actually a batch of batch
        # e.g. if sample_n = 4, the return can be like, batch = [[0,0,0,0], [2,2,2,2]]
        # Therefore we need to squeeze dim 0 first and append to the acc_batch

        acc_responses = []
        acc_indices = []
        try:
            for i, batch in enumerate(batch_iter):
                logger.info(f"--- Received Batch #{i} with {len(batch)} prompt_responses ---")
                # the batch is a batch of response lists
                for response_list in batch:
                    prompt_id = response_list.get("id", -1)
                    responses = response_list.get("data", [])
                    if len(responses) != self.sample_n:
                        logger.info(f"Number of responses {len(responses)} != {self.sample_n=}, got {responses=}")
                        raise ValueError(f"Number of responses {len(responses)} != {self.sample_n=}")
                    acc_responses.extend(responses)
                    acc_indices.extend([prompt_id for _ in range(self.sample_n)])
                    
                if len(acc_responses) >= stream_size:
                    stream_batch_size = (len(acc_responses) // stream_size) * stream_size
                    logger.info(f"[SGLangRollout] Accumulated responses {len(acc_responses)=} > {stream_size=}, process {stream_batch_size} responses")
                    to_process_batch = acc_responses[:stream_batch_size]
                    to_process_indices = acc_indices[:stream_batch_size]
                    
                    acc_responses = acc_responses[stream_batch_size:]
                    acc_indices = acc_indices[stream_batch_size:]
                    
                    # polyrl-dev
                    # response from the same prompt should have the same index
                    out = _post_process_outputs(self.processing_class, to_process_batch)
                    batch_idx = idx[to_process_indices]
                    batch_attention_mask = attention_mask[to_process_indices]
                    batch_position_ids = position_ids[to_process_indices]
                    batch_non_tensor_batch = {}
                    for key, val in non_tensor_batch.items():
                        if isinstance(val, np.ndarray) and len(val) > 0:
                            batch_non_tensor_batch[key] = val[to_process_indices]
                        else:
                            # For scalar values or empty arrays, keep as is
                            batch_non_tensor_batch[key] = val

                    # postprocessing
                    yield self.postprocess_samples(out, batch_idx, batch_attention_mask, batch_position_ids, eos_token_id, batch_non_tensor_batch)
                    # yield response
        finally:
            if acc_responses:
                out = _post_process_outputs(self.processing_class, acc_responses)
                batch_idx = idx[acc_indices]
                batch_attention_mask = attention_mask[acc_indices]
                batch_position_ids = position_ids[acc_indices]
                batch_non_tensor_batch = {}
                for key, val in non_tensor_batch.items():
                    if isinstance(val, np.ndarray) and len(val) > 0:
                        batch_non_tensor_batch[key] = val[acc_indices]
                    else:
                        # For scalar values or empty arrays, keep as is
                        batch_non_tensor_batch[key] = val

                batch_iter.close()
                acc_responses = []
                acc_indices = []
                # postprocessing
                yield self.postprocess_samples(out, batch_idx, batch_attention_mask, batch_position_ids, eos_token_id, batch_non_tensor_batch)

    def generate_sequences_remote(self, prompts: DataProto, stream_size: int = 0, **kwargs) -> Iterator[DataProto]:
        """
        Main entry point for generating sequences using the iterator wrapper.
        
        Args:
            prompts: DataProto containing input prompts
            stream_size: Size of each ibatch to yield
            **kwargs: Additional generation parameters
            
        Returns:
            Iterator yielding DataProto objects containing ibatches of generated sequences, 
            i.e. the returned number of sequences is a multiple of stream_size
        """
        prompts = prompts.to(get_torch_device().current_device())
        prompts.meta_info.update(kwargs)
        
        # ibatch size
        if stream_size == 0:
            stream_size = self.min_stream_prompts * self.sample_n
        else:
            logger.info(f"Using user specified ibatch size {stream_size}")
        
        return self._launch_generate_remote(prompts, stream_size, **kwargs)


