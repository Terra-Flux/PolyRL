from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple

# Import from the copied transfer_engine.py
from .utils import TransferEngineConfig


@dataclass
class TransferAgentConfig:
    # SGL instance's HTTP server info for identification
    sglang_http_host: str
    sglang_http_port: int
    # IP/hostname and port of the node where the senders' RPyC servers is running
    sender_rpyc_endpoints: List[Tuple[str, int]]
    engine_config: TransferEngineConfig
    num_engines: int = 1
    # Hostname/IP for the ZMQ socket on this receiver node
    zmq_bind_host: str = "0.0.0.0"


class TransferStatus(IntEnum):
    SUCCESS = 0
    FAILURE = 1
