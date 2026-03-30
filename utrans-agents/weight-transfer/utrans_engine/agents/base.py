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
import logging
import asyncio
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp
from aiohttp import web
import requests

from utrans_engine.configs.config import ReceiverConfig, ReceiverInfo
from utrans_engine.engine.tcp_engine import TCPTransferEngine
from utrans_engine.engine.shm_manager import create_shared_memory, create_tensor_from_buffer
from utrans_engine.engine.utils import get_local_ip

class BaseAgent(ABC):
    """Abstract base class for all transfer agents."""
    def __init__(self, config: Any, input_queue: mp.Queue, output_queue: mp.Queue):
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def start(self):
        """Start the agent's main processing loop."""
        pass

class BaseSender(BaseAgent, ABC):
    """Abstract base class for Sender agents."""
    
    @abstractmethod
    def update_weights(self, tensors_meta: List[Tuple[str, Tuple[List[int], str]]]):
        """Trigger weight update and distribution."""
        pass

class BaseReceiver(BaseAgent, ABC):
    """Abstract base class for Receiver agents.
    
    This base handles the common 'Receiver' tasks:
    1. HTTP control plane for notifications from Rollout Manager.
    2. TCP engine for high-speed data receipt.
    3. Bootstrapping and SHM allocation.
    4. Registration with Rollout Manager.
    """
    def __init__(self, config: ReceiverConfig, input_queue: mp.Queue, output_queue: mp.Queue):
        super().__init__(config, input_queue, output_queue)
        self.engine = TCPTransferEngine(config, config.num_parallel_streams)
        self.engine.is_receiver = True
        self.shm_obj = None
        self.buffer = None
        self.total_size = 0
        self.rpyc_conn = None
        self.bootstrapped = False

    def start(self):
        """Standard receiver startup: starts HTTP server and registers."""
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            pass

    async def _run_async(self):
        # Start HTTP server for notifications
        app = web.Application()
        app.router.add_get("/health", self.handle_health)
        app.router.add_post("/update_weights_from_agent", self.handle_update_notification)
        
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self.config.handshake_port)
        await site.start()
        
        self.logger.info(f"Receiver control plane active on port {self.config.handshake_port}")
        
        # Initial registration (bootstrap phase 1)
        self.register_with_manager()
        
        # Keep alive
        while True:
            await asyncio.sleep(3600)

    def register_with_manager(self, buffer_ptr=0, buffer_length=0):
        import rpyc
        reg_payload = {
            "host": get_local_ip(),
            "port": self.config.handshake_port,
            "mooncake_handshake_port": self.config.handshake_port
        }
        url = f"{self.config.rollout_manager_url.rstrip('/')}/register_rollout_instance"
        self.logger.info(f"Registering with manager at {url} (size={buffer_length})")
        
        try:
            resp = requests.post(url, json=reg_payload, timeout=10)
            if resp.status_code == 200:
                cfg = resp.json()
                rpyc_endpoint = cfg["weight_sender_rpyc_endpoint"]
                host, port = rpyc_endpoint.split(":")
                
                if self.rpyc_conn:
                    self.rpyc_conn.close()
                self.rpyc_conn = rpyc.connect(host, int(port), config={"allow_pickle": True})
                
                self.rpyc_conn.root.register_receiver(
                    session_ids=[self.engine.get_session_id()],
                    buffer_ptr=buffer_ptr,
                    buffer_length=buffer_length,
                    zmq_endpoint="", 
                    zmq_port=0,
                    sglang_http_host=get_local_ip(),
                    sglang_http_port=self.config.handshake_port,
                    handshake_ports=[self.config.handshake_port],
                    sender_group_index=0
                )
                self.logger.info("Registration complete.")
            else:
                self.logger.error(f"Failed to register: {resp.text}")
        except Exception as e:
            self.logger.error(f"Registration error: {e}")

    async def handle_update_notification(self, request):
        """Endpoint called by Rollout Manager when new weights are available."""
        data = await request.json()
        tensors_meta = data['tensors_meta']
        weight_version = data.get('weight_version', 0)
        bootstrap = data.get('bootstrap', False)
        
        if bootstrap:
            self.logger.info(f"Received bootstrap signal. Metadata contains {len(tensors_meta)} tensors.")
            try:
                self.setup_buffer_from_meta(tensors_meta)
                self.logger.info("Shared memory buffer allocated. Re-registering...")
                self.register_with_manager(self.engine.buffer_ptr, self.total_size)
                self.bootstrapped = True
                return web.json_response({"success": True, "message": "Bootstrapped"})
            except Exception as e:
                self.logger.error(f"Bootstrap failed: {e}")
                return web.json_response({"success": False, "message": str(e)}, status=500)

        if not self.bootstrapped:
            return web.json_response({"success": False, "message": "Not bootstrapped"}, status=400)

        self.logger.info(f"Received update v{weight_version}. Calling apply_weights...")
        success, message = await self.apply_weights(tensors_meta, weight_version, data)
        return web.json_response({"success": success, "message": message})

    def setup_buffer_from_meta(self, tensors_meta):
        """Allocate SHM based on metadata."""
        total_size = self._calculate_total_size(tensors_meta)
        self.total_size = total_size
        
        self.shm_obj, buf_addr, buffer = create_shared_memory("receiver_buffer", total_size)
        self.buffer = self._create_buffer_view(buf_addr, total_size, buffer)
        self.engine.register(buf_addr, total_size, buffer=buffer)
        self.engine.start_listener()
        self.logger.info(f"Allocated {total_size/1024/1024:.2f}MB buffer and started listeners.")

    @abstractmethod
    def _calculate_total_size(self, tensors_meta) -> int:
        pass

    @abstractmethod
    def _create_buffer_view(self, buf_addr, total_size, buffer):
        pass

    @abstractmethod
    async def apply_weights(self, tensors_meta, weight_version, extra_data) -> Tuple[bool, str]:
        """Apply received weights into the model/compute graph."""
        pass

    async def handle_health(self, request):
        return web.json_response({"status": "healthy", "bootstrapped": self.bootstrapped})
