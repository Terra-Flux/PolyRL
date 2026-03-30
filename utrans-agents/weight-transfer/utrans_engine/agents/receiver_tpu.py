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
import os
import asyncio
import numpy as np
import jax
import jax.numpy as jnp
import requests
from typing import List, Tuple, Dict, Optional

from utrans_engine.agents.base import BaseReceiver
from utrans_engine.configs.config import ReceiverConfig
from utrans_engine.engine.utils import get_local_ip

logger = logging.getLogger(__name__)

# Prevent JAX from pre-allocating all TPU memory in the sidecar
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

class TPUReceiver(BaseReceiver):
    """Receiver specialized for JAX/TPU rollouts using SHM weight updates."""
    
    def _calculate_total_size(self, tensors_meta) -> int:
        total = 0
        for name, (shape, dtype_str) in tensors_meta:
            dtype = np.dtype(dtype_str)
            numel = np.prod(shape)
            total += numel * dtype.itemsize
        return total

    def _create_buffer_view(self, buf_addr, total_size, buffer):
        # TPU sidecar works with numpy views of the SHM
        return buffer

    async def apply_weights(self, tensors_meta, weight_version, extra_data) -> Tuple[bool, str]:
        """Notify SGLang to pull from the shared SHM buffer."""
        # 1. We already have the data in self.shm_obj (SharedMemory)
        shm_name = self.shm_obj.name
        
        # 2. Build metadata with offsets for the patch
        # Matching polyrl-sglang/patch/sglang/patches.py#514 format
        offset = 0
        shm_metadata = []
        for name, (shape, dtype_str) in tensors_meta:
            numel = int(np.prod(shape))
            dtype = np.dtype(dtype_str)
            size_in_bytes = numel * dtype.itemsize
            
            shm_metadata.append({
                'name': name,
                'shape': shape,
                'dtype_str': dtype_str,
                'offset': offset,
                'size_in_bytes': size_in_bytes
            })
            offset += size_in_bytes

        # 3. Call SGLang /update_weights_from_disk
        # SGLang-JAX patch will map this SHM and perform the device_put
        sglang_payload = {
            "shm_path": shm_name,
            "tensors_meta": shm_metadata,
            "weight_version": str(weight_version)
        }
        
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, lambda: requests.post(
                f"{self.config.sglang_url.rstrip('/')}/update_weights_from_disk",
                json=sglang_payload, 
                timeout=600
            ))
            if resp.status_code == 200:
                logger.info(f"TPU sync v{weight_version} triggered successfully via SHM.")
                return True, f"TPU sync v{weight_version} triggered via SHM"
            else:
                logger.error(f"SGLang patch error: {resp.text}")
                return False, f"SGLang error: {resp.text}"
        except Exception as e:
            logger.error(f"Post-update error (TPU SHM): {e}")
            return False, f"Post-update error: {e}"

    def _reconstruct_numpy_pytree(self, tensors_meta):
        # Unused in this SHM-only path but keep for potential local checks
        offset = 0
        pytree = {}
        for name, (shape, dtype_str) in tensors_meta:
            dtype = np.dtype(dtype_str)
            numel = int(np.prod(shape))
            size = numel * dtype.itemsize
            arr = np.frombuffer(self.buffer, dtype=dtype, count=numel, offset=offset)
            pytree[name] = arr.reshape(shape)
            offset += size
        return pytree
