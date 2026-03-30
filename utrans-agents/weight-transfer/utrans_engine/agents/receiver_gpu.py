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
import logging
import torch
import torch.multiprocessing as mp
import requests
import asyncio
from typing import List, Tuple, Dict, Optional

from utrans_engine.agents.base import BaseReceiver
from utrans_engine.configs.config import ReceiverConfig
from utrans_engine.engine.shm_manager import create_tensor_from_buffer
from utrans_engine.engine.utils import MultiprocessingSerializer

logger = logging.getLogger(__name__)

def shard_tensor(name: str, tensor: torch.Tensor, tp_rank: int, tp_size: int) -> torch.Tensor:
    if tp_size <= 1:
        return tensor
    col_parallel = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "embed_tokens", "lm_head"]
    row_parallel = ["o_proj", "down_proj"]
    for layer in col_parallel:
        if layer in name:
            shard_size = tensor.shape[0] // tp_size
            return tensor[tp_rank * shard_size : (tp_rank + 1) * shard_size].contiguous()
    for layer in row_parallel:
        if layer in name:
            shard_size = tensor.shape[1] // tp_size
            return tensor[:, tp_rank * shard_size : (tp_rank + 1) * shard_size].contiguous()
    return tensor

def remap_and_stack_tensors(named_tensors):
    """Physically stack and rename tensors to match SGLang internal structure."""
    # Maps internal parameter name -> list of (shard_id, tensor)
    to_stack = {}
    final_list = []
    
    # Define stacking rules
    stacking_rules = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # 1. Categorize tensors into 'to_stack' or 'final_list'
    for name, tensor in named_tensors:
        matched = False
        for suffix, (target_name, shard_idx) in stacking_rules.items():
            if name.endswith(f".{suffix}.weight"):
                base_name = name.replace(f".{suffix}.weight", f".{target_name}.weight")
                if base_name not in to_stack:
                    to_stack[base_name] = {}
                to_stack[base_name][shard_idx] = tensor
                matched = True
                break
        if not matched:
            final_list.append((name, tensor))

    # 2. Perform physical concatenation for each stacked parameter
    for base_name, shards in to_stack.items():
        # Ensure we have all required shards (e.g., 0,1,2 for QKV)
        shard_indices = sorted(shards.keys())
        tensors_to_concat = [shards[i] for i in shard_indices]
        
        # Stack along dimension 0 (Standard for SGLang QKV/MLP stacking)
        stacked_tensor = torch.cat(tensors_to_concat, dim=0)
        final_list.append((base_name, stacked_tensor))
        
    return final_list

class GPUReceiver(BaseReceiver):
    """Receiver specialized for PyTorch/GPU rollouts."""
    
    def _calculate_total_size(self, tensors_meta) -> int:
        total = 0
        for name, (shape, dtype_str) in tensors_meta:
            dtype = getattr(torch, dtype_str)
            numel = 1
            for d in shape: numel *= d
            total += numel * torch.tensor([], dtype=dtype).element_size()
        return total

    def _create_buffer_view(self, buf_addr, total_size, buffer):
        return create_tensor_from_buffer(buf_addr, total_size, buffer)

    async def apply_weights(self, tensors_meta, weight_version, extra_data) -> Tuple[bool, str]:
        """Perform sharding and apply to SGLang."""
        full_named_tensors = self._reconstruct_full_tensors(tensors_meta)
        
        tp_size = self.config.tp_size
        serialized_shards = []
        
        # In a real cluster deployment, tp_size will be across nodes.
        # Here we assume the sidecar has access to local GPUs for the TP group.
        for rank in range(tp_size):
            rank_shard = []
            device = torch.device(f"cuda:{rank}")
            for name, tensor in full_named_tensors:
                sharded = shard_tensor(name, tensor, rank, tp_size)
                # Move to target GPU rank in sidecar to create IPC handle
                gpu_tensor = sharded.to(device, non_blocking=True)
                rank_shard.append((name, gpu_tensor))
            
            # Remap names to match SGLang internal structure
            rank_shard = remap_and_stack_tensors(rank_shard)
            serialized_shards.append(MultiprocessingSerializer.serialize(rank_shard, output_str=True))
        
        torch.cuda.synchronize()
        sglang_payload = {
            "serialized_named_tensors": serialized_shards,
            "load_format": "direct",
            "flush_cache": extra_data.get('flush_cache', True),
            "weight_version": str(weight_version)
        }
        
        try:
            # SGLang usually listens on port 30000 by default
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, lambda: requests.post(
                f"{self.config.sglang_url.rstrip('/')}/update_weights_from_tensor",
                json=sglang_payload, 
                timeout=600
            ))
            if resp.status_code == 200:
                return True, f"GPU update v{weight_version} applied to SGLang"
            else:
                return False, f"SGLang error: {resp.text}"
        except Exception as e:
            return False, f"Post-update error: {e}"

    def _reconstruct_full_tensors(self, tensors_meta):
        offset = 0
        named_tensors = []
        for name, (shape, dtype_str) in tensors_meta:
            dtype = getattr(torch, dtype_str)
            numel = 1
            for d in shape: numel *= d
            size = numel * torch.tensor([], dtype=dtype).element_size()
            tensor = self.buffer[offset : offset + size].view(dtype).view(*shape)
            named_tensors.append((name, tensor))
            offset += size
        return named_tensors
