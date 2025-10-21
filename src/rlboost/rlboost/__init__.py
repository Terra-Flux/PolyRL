"""rlboost public API.

Usage:
    import rlboost as rlb
    rlb.start_transfer_agent(...)
    rlb.FSDPInterface(...)
"""

from .weight_transfer import (
    start_transfer_agent,
    spawn_rollout_manager,
    register_weight_senders,
    FSDPInterface,
)

from .weight_transfer.config import (
    MooncakeTransferEngineConfig,
    TransferAgentConfig,
    TransferStatus,
    ReceiverInfo,
)

__all__ = [
    "start_transfer_agent",
    "spawn_rollout_manager",
    "register_weight_senders",
    "FSDPInterface",
    "MooncakeTransferEngineConfig",
    "TransferAgentConfig",
    "TransferStatus",
    "ReceiverInfo",
]

