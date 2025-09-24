from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Union, List, Tuple

# Import from the copied transfer_engine.py
from .transfer_engine import MooncakeTransferEngineConfig


@dataclass
class TransferAgentConfig:
    # SGL instance's HTTP server info for identification
    sglang_http_host: str
    sglang_http_port: int
    # IP/hostname and port of the node where the senders' RPyC servers is running
    sender_rpyc_endpoints: List[Tuple[str, int]]
    # Optional path to mooncake config file or the config object itself
    mooncake_config: Optional[Union[MooncakeTransferEngineConfig, str]] = None
    # Hostname/IP for the ZMQ socket on this receiver node
    zmq_bind_host: str = "0.0.0.0"
    # Port for mooncake P2P handshake
    mooncake_handshake_port: int = None


class TransferStatus(IntEnum):
    SUCCESS = 0
    FAILURE = 1
