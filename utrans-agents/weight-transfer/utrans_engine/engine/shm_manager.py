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
import os
import mmap
import ctypes
import logging
from typing import Tuple, Optional
from multiprocessing import shared_memory

logger = logging.getLogger(__name__)

def create_shared_memory(name_hint: str, size: int) -> Tuple[shared_memory.SharedMemory, int, memoryview]:
    """Create shared memory and return (shm_obj, ptr, memview)."""
    # Create a new shared memory segment
    shm = shared_memory.SharedMemory(create=True, size=size)
    
    # Get the buffer address for the engine using a more robust method
    try:
        # Use addressof(c_char.from_buffer) as it's more reliable than void_p.value
        # for memoryviews in some Python versions.
        buf_addr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
    except Exception as e:
        logger.debug(f"Address resolution failed: {e}")
        buf_addr = 0 # Fallback
    
    # Try mlock for performance (Linux/Darwin)
    if buf_addr != 0:
        try:
            libc = ctypes.CDLL(ctypes.util.find_library('c'))
            libc.mlock(ctypes.c_void_p(buf_addr), ctypes.c_size_t(size))
        except Exception as e:
            logger.debug(f"Mlock failed: {e}")
        
    return shm, buf_addr, shm.buf

def create_tensor_from_buffer(buf_addr: int, size: int, buffer: memoryview):
    """Create a torch tensor from a raw buffer."""
    # Darwin safety fix
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.cudart().cudaHostRegister(buf_addr, size, 0)
    except:
        pass
        
    tensor = torch.frombuffer(buffer, dtype=torch.uint8)
    return tensor
