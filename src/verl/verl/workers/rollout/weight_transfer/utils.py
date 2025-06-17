from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Union

from .transfer_engine import MooncakeTransferEngineConfig


@dataclass
class TransferAgentConfig:
    trainer_global_rank: int
    trainer_world_size: int
    mooncake_config: Optional[Union[MooncakeTransferEngineConfig, str]] = None
    rpyc_bind_port: int = 18861
    mooncake_handshake_port: int = None


class TransferStatus(IntEnum):
    SUCCESS = 0
    FAILURE = 1
