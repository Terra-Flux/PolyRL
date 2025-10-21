"""
Configuration classes for weight transfer system.

This module contains all configuration classes used in the weight transfer pipeline,
including transfer engine configs, agent configs, and manager configs.
"""

import json
import os
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

# Transfer Engine Configuration
@dataclass
class MooncakeTransferEngineConfig:
    """Configuration for Mooncake transfer engine."""
    local_hostname: str
    protocol: str
    device_name: str
    handshake_port: int

    @staticmethod
    def from_file(file_path: str) -> "MooncakeTransferEngineConfig":
        """Load configuration from JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeTransferEngineConfig(
            local_hostname=config.get("local_hostname", None),
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            handshake_port=config.get("handshake_port", None),
        )

    @staticmethod
    def load_from_env() -> "MooncakeTransferEngineConfig":
        """Load configuration from environment variable."""
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeTransferEngineConfig.from_file(config_file_path)


# Transfer Agent Configuration
@dataclass
class TransferAgentConfig:
    """Configuration for transfer agent."""
    trainer_global_rank: int
    trainer_world_size: int
    mooncake_config: List[MooncakeTransferEngineConfig]
    num_mooncake_engines_per_group: int = 1
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
    "MooncakeTransferEngineConfig",
    "TransferAgentConfig", 
    "TransferStatus",
    "ReceiverInfo",
]
