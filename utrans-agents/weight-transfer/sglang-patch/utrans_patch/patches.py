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
import types
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from aiohttp import web
from typing import Dict, List, Tuple, Optional, Any
from multiprocessing import shared_memory
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from contextlib import nullcontext

from .autopatch import BasePatch

logger = logging.getLogger(__name__)

@dataclass
class UpdateWeightsFromDiskReqInput:
    shm_path: str
    tensors_meta: List[Dict[str, Any]]
    flush_cache: bool = True
    abort_all_requests: bool = False
    weight_version: Optional[str] = None

@dataclass
class UpdateWeightsFromDiskReqOutput:
    success: bool
    message: str
    
class IOStructPatch(BasePatch):
    def apply(self) -> bool:
        try:
            from sgl_jax.srt.managers import io_struct

            if hasattr(io_struct, 'UpdateWeightsFromDiskReqInput'):
                return True

            io_struct.UpdateWeightsFromDiskReqInput = UpdateWeightsFromDiskReqInput
            io_struct.UpdateWeightsFromDiskReqOutput = UpdateWeightsFromDiskReqOutput

            return True
        except Exception as e:
            logger.error(f"IOStructPatch failed: {e}")
            return False

class ModelRunnerPatch(BasePatch):
    def apply(self) -> bool:
        try:
            from sgl_jax.srt.model_executor.model_runner import ModelRunner
            from sgl_jax.srt.utils.weight_utils import WeightLoader
            
            if self._is_patched(ModelRunner.initialize_jit, "init"): return True

            original_initialize_jit = ModelRunner.initialize_jit

            def patched_initialize_jit(self):
                original_initialize_jit(self)
                # Capture model structure for tree swap
                _, model_state = nnx.split(self.model)
                self.model_state_leaves, self.model_state_def = jax.tree_util.tree_flatten(model_state)
                
                # Re-jit run_model to use the dynamic state leaves
                model_def, _ = nnx.split(self.model)
                @partial(jax.jit, donate_argnames=["token_to_kv_pool"], static_argnames=["model_state_def"])
                def jitted_run_model(model_def, model_state_def, model_state_leaves, forward_batch, token_to_kv_pool, logits_metadata):
                    model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
                    model = nnx.merge(model_def, model_state)
                    return model(forward_batch, token_to_kv_pool, logits_metadata)

                self.jitted_run_model = lambda fb, lm: jitted_run_model(model_def, self.model_state_def, self.model_state_leaves, fb, self.token_to_kv_pool, lm)
                logger.info("ModelRunner: High-performance dynamic weight path initialized.")

            def update_weights_from_disk(self, req: UpdateWeightsFromDiskReqInput):
                try:
                    # 0. Cache SHM and Resolved Mappings
                    if not hasattr(self, "_shm_cache"): self._shm_cache = {}
                    if not hasattr(self, "_resolved_mappings"): self._resolved_mappings = []
                    
                    if req.shm_path not in self._shm_cache:
                        self._shm_cache = {req.shm_path: shared_memory.SharedMemory(name=req.shm_path)}
                    shm = self._shm_cache[req.shm_path]
                    
                    # 1. Resolve mappings and pre-compute sharding (Cold path)
                    if not self._resolved_mappings:
                        params = nnx.state(self.model)
                        raw_mappings = {}
                        for attr in dir(self.model):
                            if attr.startswith("_create_") and attr.endswith("_weight_mappings"):
                                raw_mappings = getattr(self.model, attr)()
                                break
                        
                        loader = WeightLoader(self.model, self.model_config, self.mesh)
                        for hf_key, mapping in raw_mappings.items():
                            try:
                                path = mapping.target_path if isinstance(mapping.target_path, str) else mapping.target_path[0]
                                var = loader._get_param(params, path)
                                # Pre-compute sharding and target dtype once
                                sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(*mapping.sharding))
                                target_dtype = var.value.dtype
                                self._resolved_mappings.append((hf_key, var, mapping, sharding, target_dtype))
                            except: pass
                        logger.info(f"SGLang-JAX: Resolved and cached {len(self._resolved_mappings)} tensors with pre-computed sharding.")

                    # 2. Update Loop (Hot path)
                    update_dict = {item['name']: item for item in req.tensors_meta}
                    start_time = time.perf_counter()
                    updated_count = 0
                    
                    for hf_key, var, mapping, sharding, target_dtype in self._resolved_mappings:
                        if hf_key in update_dict:
                            m = update_dict[hf_key]
                            # Zero-copy SHM view
                            arr = np.frombuffer(shm.buf, dtype=np.dtype(m['dtype_str']), count=int(np.prod(m['shape'])), offset=m['offset']).reshape(m['shape'])
                            
                            # CPU Transpose (pointer swap)
                            if mapping.transpose and arr.ndim == 2: arr = arr.T
                            
                            # lm_head scaling
                            if "lm_head" in hf_key and hasattr(self.model_config.hf_config, "output_multiplier_scale"):
                                arr = arr.astype(np.float32) * self.model_config.hf_config.output_multiplier_scale
                            
                            # Direct HBM Put
                            var.value = jax.device_put(arr, sharding).astype(target_dtype)
                            updated_count += 1

                    # 3. Atomic HBM Swap
                    # We use the Variable references to update the actual model state leaves
                    model_state = nnx.state(self.model)
                    leaves, _ = jax.tree_util.tree_flatten(model_state)
                    self.model_state_leaves = leaves
                    jax.block_until_ready(self.model_state_leaves)

                    logger.info(f"SGLang-JAX: Applied v{req.weight_version} ({updated_count} tensors) in {time.perf_counter()-start_time:.3f}s")
                    return True, "Success"
                except Exception as e:
                    logger.error(f"SGLang-JAX Update Failed: {e}")
                    return False, str(e)

            ModelRunner.initialize_jit = patched_initialize_jit
            ModelRunner.update_weights_from_disk = update_weights_from_disk
            return True
        except Exception as e:
            logger.error(f"ModelRunnerPatch failed: {e}")
            return False

class SchedulerPatch(BasePatch):
    def apply(self) -> bool:
        try:
            from sgl_jax.srt.managers.scheduler import Scheduler
            original_init = Scheduler.__init__
            if self._is_patched(original_init, "init"): return True

            def patched_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self._request_dispatcher._mapping.append((UpdateWeightsFromDiskReqInput, self.update_weights_from_disk))

            def update_weights_from_disk(self, req: UpdateWeightsFromDiskReqInput):
                mw = self.tp_worker
                model_runner = mw.model_runner if hasattr(mw, "model_runner") else mw.worker.model_runner
                success, message = model_runner.update_weights_from_disk(req)
                if success:
                    # must flush cache after weight update
                    if req.flush_cache:
                        logger.info("Skip flush_cache because we retracted on pause!")
                        # flush_cache_success = self.flush_cache()[0]
                        # assert flush_cache_success, "Cache flush failed after updating weights"
                else:
                    logger.error(message)
                return UpdateWeightsFromDiskReqOutput(success, message)

            self._mark_as_patched(original_init, "init")
            Scheduler.__init__ = patched_init
            Scheduler.update_weights_from_disk = update_weights_from_disk
            return True
        except Exception as e:
            logger.error(f"SchedulerPatch failed: {e}")
            return False

class TokenizerManagerPatch(BasePatch):
    def apply(self) -> bool:
        try:
            from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager, _Communicator
            from sgl_jax.srt.managers import io_struct
            original_init = TokenizerManager.__init__
            if self._is_patched(original_init, "init"): return True

            def patched_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                self.update_weights_from_disk_communicator = _Communicator(self.send_to_scheduler, self.server_args.dp_size)
                self._result_dispatcher._mapping.append((UpdateWeightsFromDiskReqOutput, self.update_weights_from_disk_communicator.handle_recv))

            async def update_weights_from_disk(
                self: TokenizerManager,
                obj: UpdateWeightsFromDiskReqInput, 
                request=None,
            ):
                if obj.abort_all_requests:
                    self.abort_request(abort_all=True)
                
                # pause generation and retract requests to waiting_queue for recompute KV cache    
                await self.pause_generation(io_struct.PauseGenerationReqInput("retract"))

                # NOTE: only keep the first one in SPMD
                result = (await self.update_weights_from_disk_communicator(obj))[0]
                
                await self.continue_generation(io_struct.ContinueGenerationReqInput())
                
                return result.success, result.message

            self._mark_as_patched(original_init, "init")
            TokenizerManager.__init__ = patched_init
            TokenizerManager.update_weights_from_disk = update_weights_from_disk
            return True
        except Exception as e:
            logger.error(f"TokenizerManager failed: {e}")
            return False

class HttpServerPatch(BasePatch):
    def apply(self) -> bool:
        try:
            from sgl_jax.srt.entrypoints import http_server
            from fastapi import Request
            from fastapi.responses import JSONResponse
            app = http_server.app
            @app.post("/update_weights_from_disk")
            async def update_weights_from_disk(
                obj: UpdateWeightsFromDiskReqInput,
                request: Request
            ):
                if http_server._global_state is None: return JSONResponse({"success": False}, status_code=503)
                success, message = await http_server._global_state.tokenizer_manager.update_weights_from_disk(
                    obj, request
                )
                return JSONResponse({"success": success, "message": message}, status_code=200 if success else HTTPStatus.BAD_REQUEST)
            return True
        except Exception as e:
            logger.error(f"HttpServerPatch failed: {e}")
            return False
