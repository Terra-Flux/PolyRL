"""
Configuration classes for weight transfer system.

This module contains all configuration classes used in the weight transfer pipeline,
including transfer engine configs, agent configs, and manager configs.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List

from .utils import TransferEngineConfig


# Transfer Agent Configuration
@dataclass
class TransferAgentConfig:
    """Configuration for transfer agent."""
    trainer_global_rank: int
    trainer_world_size: int
    engine_configs: List[TransferEngineConfig]
    num_engines_per_group: int = 1
    rpyc_bind_port: int = 18861


class TransferStatus(IntEnum):
    """Status codes for transfer operations."""
    SUCCESS = 0
    FAILURE = 1

# Receiver Information (for internal use)
@dataclass
class ReceiverInfo:
    """Information about a registered receiver."""
    session_ids: List[str]
    buffer_ptr: int
    buffer_length: int
    zmq_endpoint: str
    zmq_port: int
    sglang_http_host: str
    sglang_http_port: int
    handshake_ports: List[int]
    sender_group_index: int


__all__ = [
    "TransferEngineConfig",
    "TransferAgentConfig", 
    "TransferStatus",
    "ReceiverInfo",
]
