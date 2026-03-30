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
import asyncio
import logging
import queue
import threading
import torch
import rpyc
from typing import Dict, List, Tuple, Optional
from rpyc.utils.server import ThreadedServer
import os
import time

from utrans_engine.agents.base import BaseSender
from utrans_engine.configs.config import SenderConfig, ReceiverInfo, TransferStatus
from utrans_engine.engine.tcp_engine import TCPTransferEngine
from utrans_engine.engine.shm_manager import create_shared_memory, create_tensor_from_buffer
from utrans_engine.engine.utils import get_local_ip

logger = logging.getLogger(__name__)

class Sender(BaseSender):
    """General implementation of a sender agent."""
    def __init__(self, config: SenderConfig, input_queue: queue.Queue, output_queue: queue.Queue, manager_url: Optional[str] = None):
        super().__init__(config, input_queue, output_queue)
        self.receivers: Dict[str, ReceiverInfo] = {}
        # Track the version currently held by each receiver
        self.receiver_versions: Dict[str, int] = {}
        self.receiver_failures: Dict[str, int] = {}
        self.weight_version = 0
        self.tensors_meta = None
        self.engine = TCPTransferEngine(config, config.num_parallel_streams)
        self.buffer = None
        self.shm_obj = None
        self.manager_url = manager_url
        
        # Start RPyC for control channel
        self.rpyc_server = None
        self.rpyc_thread = None

    def start(self):
        """Start the sender's main loop and control server."""
        # Initial metadata setup
        meta_sizes = self.input_queue.get()
        self.tensors_meta = self.input_queue.get()
        
        total_size = sum(s for n, s in meta_sizes)
        self.buffer_length = total_size
        self.shm_obj, buf_addr, buffer = create_shared_memory("weight_buffer", total_size)
        self.buffer = create_tensor_from_buffer(buf_addr, total_size, buffer)
        
        # On Linux, SharedMemory objects have a file descriptor under /dev/shm
        # We need the FD for sendfile optimization
        try:
            shm_fd = os.open(f"/dev/shm/{self.shm_obj.name}", os.O_RDONLY)
            self.engine.register_memfd(shm_fd, total_size)
        except Exception as e:
            logger.warning(f"Could not open SHM FD for sendfile: {e}. Performance may be degraded.")
        
        # Return name and size (picklable)
        self.output_queue.put((self.shm_obj.name, total_size))
        
        # Start RPyC thread
        self.rpyc_server = ThreadedServer(
            SenderService(self), 
            port=self.config.rpyc_bind_port,
            protocol_config={"allow_pickle": True}
        )
        self.rpyc_thread = threading.Thread(target=self.rpyc_server.start, daemon=True)
        self.rpyc_thread.start()
        
        logger.info(f"Sender started on {get_local_ip()}:{self.config.rpyc_bind_port}")
        self.event_loop()

    def event_loop(self):
        while True:
            try:
                cmd = self.input_queue.get(timeout=1.0)
                if cmd == "update_weights":
                    self.weight_version += 1
                    logger.info(f"Trainer signaled update. New version: {self.weight_version}")
                    self.output_queue.put("completed")
                    self.check_and_update_receivers()
            except queue.Empty:
                # Periodic check for new receivers that need current weights
                if self.receivers:
                    self.check_and_update_receivers()

    def check_and_update_receivers(self):
        """Find receivers with out-of-date weights and push the current version."""
        if self.weight_version == 0:
            return

        to_update = []
        for endpoint in self.receivers:
            if self.receiver_versions.get(endpoint, 0) < self.weight_version:
                to_update.append(endpoint)
        
        if not to_update:
            return

        logger.info(f"Detected {len(to_update)} receivers needing version {self.weight_version}")
        
        completed_endpoints = []
        dropped_endpoints = []
        
        for endpoint in to_update:
            info = self.receivers[endpoint]
            logger.info(f"Pushing weights to {endpoint} (v{self.weight_version})...")

            try:
                start_time = time.perf_counter()
                batch_id = self.engine.transfer_submit_write(
                    info.session_ids[0], 0, info.buffer_ptr, self.buffer_length
                )

                if batch_id < 0:
                    logger.error(f"Failed to submit transfer to {endpoint}")
                    continue

                while self.engine.transfer_check_status(batch_id) == 0:
                    time.sleep(0.001)

                if self.engine.transfer_check_status(batch_id) == 1:
                    duration = time.perf_counter() - start_time
                    bw = (self.buffer_length / 1024 / 1024) / duration if duration > 0 else 0
                    logger.info(f"Sync complete for {endpoint} in {duration:.3f}s ({bw:.2f}MB/s)")

                    self.receiver_versions[endpoint] = self.weight_version
                    self.receiver_failures[endpoint] = 0 # Reset on success
                    completed_endpoints.append(endpoint)
                else:
                    logger.error(f"Sync failed for {endpoint}")
                    self.receiver_failures[endpoint] = self.receiver_failures.get(endpoint, 0) + 1
            except Exception as e:
                logger.error(f"Error during push to {endpoint}: {e}")
                self.receiver_failures[endpoint] = self.receiver_failures.get(endpoint, 0) + 1
            
            if self.receiver_failures.get(endpoint, 0) >= 5:
                logger.error(f"Receiver {endpoint} failed 5 times. Dropping it.")
                dropped_endpoints.append(endpoint)

        for endpoint in dropped_endpoints:
            if endpoint in self.receivers:
                del self.receivers[endpoint]
            if endpoint in self.receiver_versions:
                del self.receiver_versions[endpoint]
            if endpoint in self.receiver_failures:
                del self.receiver_failures[endpoint]


        # Notify the manager to trigger sidecar sharding/application
        if self.manager_url and completed_endpoints:
            try:
                import requests
                payload = {
                    "instance_endpoints": completed_endpoints,
                    "weight_version": self.weight_version,
                    "tensors_meta": self.tensors_meta,
                    "bootstrap": False
                }
                requests.post(f"{self.manager_url.rstrip('/')}/update_weights", json=payload, timeout=10)
            except Exception as e:
                logger.error(f"Failed to notify manager: {e}")

    def update_weights(self, tensors_meta: List[Tuple[str, Tuple[List[int], str]]]):
        """Implementation of abstract method. The real logic is in check_and_update_receivers."""
        pass

    def register_receiver(self, 
                          instance_id: str,
                          session_ids: List[str],
                          buffer_ptr: int,
                          buffer_length: int,
                          zmq_endpoint: str,
                          zmq_port: int,
                          sglang_http_host: str,
                          sglang_http_port: int,
                          handshake_ports: List[int],
                          sender_group_index: int = 0):
        self.receivers[instance_id] = ReceiverInfo(
            session_ids=session_ids,
            buffer_ptr=buffer_ptr,
            buffer_length=buffer_length,
            zmq_endpoint=zmq_endpoint,
            zmq_port=zmq_port,
            sglang_http_host=sglang_http_host,
            sglang_http_port=sglang_http_port,
            handshake_ports=handshake_ports,
            sender_group_index=sender_group_index
        )
        logger.info(f"Registered receiver: {instance_id} (size: {buffer_length})")

        # Automatically trigger bootstrap if receiver is new (length 0)
        if buffer_length == 0 and self.manager_url:
            logger.info(f"Triggering bootstrap for new receiver: {instance_id}")
            try:
                import requests
                payload = {
                    "instance_endpoints": [instance_id],
                    "weight_version": 0,
                    "tensors_meta": self.tensors_meta,
                    "bootstrap": True
                }
                requests.post(f"{self.manager_url.rstrip('/')}/update_weights", json=payload, timeout=5)
            except Exception as e:
                logger.error(f"Failed to trigger bootstrap: {e}")


class SenderService(rpyc.Service):
    def __init__(self, agent: Sender):
        self.agent = agent

    def exposed_register_receiver(self, *args, **kwargs):
        from rpyc.utils.classic import obtain
        session_ids = obtain(kwargs.get('session_ids'))
        buffer_ptr = kwargs.get('buffer_ptr')
        buffer_length = kwargs.get('buffer_length')
        zmq_endpoint = kwargs.get('zmq_endpoint')
        zmq_port = kwargs.get('zmq_port')
        sglang_http_host = kwargs.get('sglang_http_host')
        sglang_http_port = kwargs.get('sglang_http_port')
        handshake_ports = obtain(kwargs.get('handshake_ports'))
        sender_group_index = kwargs.get('sender_group_index', 0)

        instance_id = f"{sglang_http_host}:{sglang_http_port}"
        self.agent.register_receiver(
            instance_id,
            session_ids,
            buffer_ptr,
            buffer_length,
            zmq_endpoint,
            zmq_port,
            sglang_http_host,
            sglang_http_port,
            handshake_ports,
            sender_group_index
        )
        return True
