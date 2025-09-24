from dataclasses import dataclass
from enum import IntEnum
from typing import List

from .transfer_engine import MooncakeTransferEngineConfig


@dataclass
class TransferAgentConfig:
    trainer_global_rank: int
    trainer_world_size: int
    mooncake_config: List[MooncakeTransferEngineConfig]
    num_mooncake_engines_per_group: int = 1
    rpyc_bind_port: int = 18861


class TransferStatus(IntEnum):
    SUCCESS = 0
    FAILURE = 1
