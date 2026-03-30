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
import types
from typing import Dict, List, Callable, Any

logger = logging.getLogger(__name__)

class BasePatch:
    def __init__(self):
        pass

    def _mark_as_patched(self, func: Callable, identifier: str):
        marker = f"__utrans_patched_{identifier}__"
        setattr(func, marker, True)

    def _is_patched(self, func: Callable, identifier: str) -> bool:
        marker = f"__utrans_patched_{identifier}__"
        return hasattr(func, marker)

    def apply(self) -> bool:
        raise NotImplementedError

class PatchManager:
    def __init__(self):
        self.patches: List[BasePatch] = []

    def register(self, patch: BasePatch):
        self.patches.append(patch)
        return self

    def apply_all(self) -> Dict[str, bool]:
        results = {}
        for patch in self.patches:
            patch_name = patch.__class__.__name__
            try:
                success = patch.apply()
                results[patch_name] = success
                if success:
                    logger.info(f"Successfully applied patch: {patch_name}")
            except Exception as e:
                logger.error(f"Failed to apply patch {patch_name}: {e}")
                results[patch_name] = False
        return results

def apply_patches():
    from .patches import (
        IOStructPatch,
        ModelRunnerPatch,
        SchedulerPatch,
        TokenizerManagerPatch,
        HttpServerPatch,
    )

    manager = PatchManager()
    manager.register(IOStructPatch())
    manager.register(ModelRunnerPatch())
    manager.register(SchedulerPatch())
    manager.register(TokenizerManagerPatch())
    manager.register(HttpServerPatch())

    results = manager.apply_all()
    return all(results.values())

# --- Automatic Cross-Process Patching via wrapt ---
try:
    from wrapt.importer import when_imported
    
    @when_imported("sgl_jax")
    def _patch_sglang(module: types.ModuleType) -> None:

        from .patches import (
            IOStructPatch,
            ModelRunnerPatch,
            SchedulerPatch,
            TokenizerManagerPatch,
            HttpServerPatch,
        )

        # logger.info("Auto-applying local patches to sgl_jax...")

        manager = PatchManager()
        manager.register(IOStructPatch())
        manager.register(ModelRunnerPatch())
        manager.register(SchedulerPatch())
        manager.register(TokenizerManagerPatch())
        manager.register(HttpServerPatch())

        results = manager.apply_all()
        return all(results.values())

    logger.info("UTrans-Engine: Registered automatic wrapt hooks for SGLang-JAX.")
except ImportError:
    logger.warning("wrapt not found. Automatic cross-process patching might fail.")
except Exception as e:
    logger.error(f"Failed to register wrapt hooks: {e}")
