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
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple, Union

class TransferStatus(IntEnum):
    """Status codes for transfer operations."""
    SUCCESS = 0
    FAILURE = 1

@dataclass
class TransferEngineConfig:
    """Configuration for low-level transfer engine."""
    local_hostname: str
    handshake_port: int

@dataclass
class TransferAgentConfig:
    """Base configuration for transfer agents."""
    local_hostname: str
    handshake_port: int
    num_parallel_streams: int = 64

@dataclass
class SenderConfig(TransferAgentConfig):
    """Configuration for sender transfer agent."""
    trainer_global_rank: int = 0
    trainer_world_size: int = 1
    rpyc_bind_port: int = 18861

@dataclass
class ReceiverConfig(TransferAgentConfig):
    """Configuration for receiver transfer agent."""
    receiver_rank: int = 0
    receiver_world_size: int = 1
    zmq_bind_host: str = "0.0.0.0"
    sglang_url: str = "http://127.0.0.1:30000"
    tp_size: int = 1
    rollout_manager_url: str = "http://127.0.0.1:8000"

@dataclass
class ReceiverInfo:
    """Information about a registered receiver used by the sender."""
    session_ids: List[str]
    buffer_ptr: int
    buffer_length: int
    zmq_endpoint: str
    zmq_port: int
    sglang_http_host: str
    sglang_http_port: int
    handshake_ports: List[int]
    sender_group_index: int
