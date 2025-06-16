import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MooncakeTransferEngineConfig:
    local_hostname: str
    protocol: str
    device_name: str
    handshake_port: int

    @staticmethod
    def from_file(file_path: str) -> "MooncakeTransferEngineConfig":
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeTransferEngineConfig(
            local_hostname=config.get("local_hostname", None),
            # polyrl-dev
            # NOTE(yongji): Here we use TCP transport by default
            protocol=config.get("protocol", "tcp"),
            device_name=config.get("device_name", ""),
            handshake_port=config.get("handshake_port", None),
        )

    @staticmethod
    def load_from_env() -> "MooncakeTransferEngineConfig":
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv("MOONCAKE_CONFIG_PATH")
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set."
            )
        return MooncakeTransferEngineConfig.from_file(config_file_path)


class MooncakeTransferEngine:

    def __init__(self, config: Optional[MooncakeTransferEngineConfig] = None, config_path: Optional[str] = None):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
            ) from e

        self.engine = TransferEngine()

        try:
            if config is not None and config_path is not None:
                raise ValueError("Only one of config or config_path should be provided")
            if config is not None:
                self.config = config
            elif config_path is not None:
                self.config = MooncakeTransferEngineConfig.from_file(config_path)
            else:
                self.config = MooncakeTransferEngineConfig.load_from_env()
            logger.info("Mooncake Configuration loaded successfully.")
        except ValueError as e:
            logger.error(e)
            raise
        except Exception as exc:
            logger.error("An error occurred while loading the configuration: %s", exc)
            raise

        # For P2P handshake, use hostname:port as session ID
        if self.config.handshake_port:
            self.session_id = f"{self.config.local_hostname}:{self.config.handshake_port}"
        else:
            session_suffix = "_" + str(uuid.uuid4())
            self.session_id = self.config.local_hostname + session_suffix

        self.initialize(
            self.session_id,
            "P2PHANDSHAKE",
            self.config.protocol,
            self.config.device_name,
        )

    def register(self, ptr, length):
        self.engine.register_memory(ptr, length)

    def deregister(self, ptr):
        self.engine.unregister_memory(ptr)

    def initialize(
        self,
        local_hostname: str,
        metadata_server: str,
        protocol: str,
        device_name: str,
    ) -> None:
        """Initialize the mooncake instance."""
        self.engine.initialize(local_hostname, metadata_server, protocol, device_name)

    def transfer_sync(
        self, session_id: str, buffer: int, peer_buffer_address: int, length: int
    ) -> int:
        """Synchronously transfer data to the specified address."""

        ret = self.engine.transfer_sync_write(
            session_id, buffer, peer_buffer_address, length
        )
        if ret < 0:
            logger.error("Transfer Return Error")
            raise Exception("Transfer Return Error")
        return ret

    def get_hostname(self):
        return self.config.local_hostname

    def get_session_id(self):
        return self.session_id
    
    def get_rpc_port(self):
        """Get the RPC port for P2P handshake mode."""
        return self.config.handshake_port if self.config.handshake_port else None