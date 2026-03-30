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
from abc import ABC, abstractmethod
from typing import List, Optional

class BaseTransferEngine(ABC):
    """Abstract base class for all transfer engines."""
    
    @abstractmethod
    def register(self, ptr: int, length: int):
        """Register a buffer for receiving data."""
        pass
    
    @abstractmethod
    def register_memfd(self, memfd: int, length: int):
        """Register a memfd for sending data."""
        pass
    
    @abstractmethod
    def deregister(self, ptr: int):
        """Deregister a buffer."""
        pass
    
    @abstractmethod
    def start_listener(self):
        """Start listening for incoming transfers."""
        pass
    
    @abstractmethod
    def transfer_submit_write(self, session_id: str, buffer: int, peer_buffer_address: int, length: int) -> int:
        """Submit an asynchronous write to a remote buffer."""
        pass
    
    @abstractmethod
    def transfer_check_status(self, batch_id: int) -> int:
        """Check the status of a transfer."""
        pass

    @abstractmethod
    def get_session_id(self) -> str:
        """Get the unique session ID of this engine."""
        pass
    
    @abstractmethod
    def get_rpc_port(self) -> int:
        """Get the listener RPC/Handshake port."""
        pass
