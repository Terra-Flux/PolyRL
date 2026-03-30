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
import logging
import os
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from .base import BaseTransferEngine
from utrans_engine.configs.config import TransferEngineConfig

logger = logging.getLogger(__name__)

class TCPTransferEngine(BaseTransferEngine):
    """TCP-based implementation of the transfer engine."""
    
    def __init__(self, config: TransferEngineConfig, num_threads: int = 6):
        self.config = config
        
        self.buffer_ptr: Optional[int] = None
        self.buffer_length: Optional[int] = None
        self.buffer_memview: Optional[memoryview] = None
        self.listener_threads = []
        self.listener_sockets = []
        self.listener_ports = []
        
        if self.config.handshake_port:
            self.session_id = f"{self.config.local_hostname}:{self.config.handshake_port}"
        else:
            session_suffix = "_" + str(uuid.uuid4())
            self.session_id = self.config.local_hostname + session_suffix
        
        self.connections: Dict[str, socket.socket] = {}
        self.connection_lock = threading.Lock()
        self.num_parallel_streams = num_threads
        self.transfer_executor = ThreadPoolExecutor(max_workers=self.num_parallel_streams * 2)
        self.pending_transfers: Dict[int, Dict] = {}
        self.next_batch_id = 1
        self.batch_id_lock = threading.Lock()
        self.is_receiver = False
        
        # Performance tunings
        self.rcvbuf_size = 16 * 1024 * 1024
        self.sndbuf_size = 16 * 1024 * 1024
        self.chunk_size = 64 * 1024 * 1024
        
        self.use_zerocopy = os.environ.get('TCP_ZEROCOPY', '0') == '1'

    def register(self, ptr: int, length: int, buffer: Optional[memoryview] = None):
        """Register buffer for receive operations (receiver side)"""
        self.buffer_ptr = ptr
        self.buffer_length = length
        
        if buffer is not None:
            self.buffer_memview = buffer
        else:
            import ctypes
            # Fallback to address-based view if buffer not provided
            buf = (ctypes.c_byte * length).from_address(ptr)
            self.buffer_memview = memoryview(buf)
            
        logger.info(f"TCPTransferEngine registered buffer: addr={ptr:#x}, size={length/1024/1024:.2f}MB")
    
    def register_memfd(self, memfd: int, length: int):
        self.memfd = memfd
        self.buffer_length = length
        logger.info(f"TCPTransferEngine registered memfd: fd={memfd}, size={length/1024/1024:.2f}MB")

    def deregister(self, ptr: int):
        self.buffer_memview = None
        self.buffer_ptr = None
        self.buffer_length = None
        self.memfd = None

    def start_listener(self):
        if len(self.listener_threads) > 0:
            return
        
        for i in range(self.num_parallel_streams):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf_size)
            sock.bind(('', 0))
            port = sock.getsockname()[1]
            sock.listen(256)
            
            self.listener_sockets.append(sock)
            self.listener_ports.append(port)
            
            thread = threading.Thread(target=self._accept_connections, args=(sock, i), daemon=True)
            thread.start()
            self.listener_threads.append(thread)
        logger.info(f"TCPTransferEngine started {self.num_parallel_streams} listeners on ports {self.listener_ports}")

    def _accept_connections(self, sock: socket.socket, listener_idx: int):
        while True:
            try:
                conn, addr = sock.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf_size)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.sndbuf_size)
                thread_id = f"{addr[0]}:{addr[1]}-L{listener_idx}"
                self.transfer_executor.submit(self._receive_data, conn, thread_id)
            except Exception as e:
                if sock:
                    logger.error(f"Accept error on listener {listener_idx}: {e}")
                break

    def _receive_data(self, conn: socket.socket, thread_id: str):
        try:
            if hasattr(socket, 'TCP_QUICKACK'):
                try: conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
                except: pass
            
            header = conn.recv(16)
            if len(header) < 16:
                logger.error(f"[{thread_id}] Short header: expected 16, got {len(header)}")
                return
            
            offset = int.from_bytes(header[:8], 'little')
            length = int.from_bytes(header[8:16], 'little')
            
            if offset + length > self.buffer_length:
                logger.error(f"[{thread_id}] OOB access: {offset}+{length} > {self.buffer_length}")
                return
            
            view = self.buffer_memview[offset:offset + length]
            received = 0
            while received < length:
                chunk_size = min(self.chunk_size, length - received)
                n = conn.recv_into(view[received:received + chunk_size], chunk_size)
                if n == 0: raise RuntimeError("Connection closed prematurely")
                received += n
            
        except Exception as e:
            logger.error(f"[{thread_id}] Receive error: {e}")
        finally:
            conn.close()

    def _create_connection(self, target_host: str, target_port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.sndbuf_size)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.rcvbuf_size)
        sock.connect((target_host, target_port))
        return sock

    def _send_data_chunk(self, sock: socket.socket, target_host: str, target_port: int,
                         local_offset: int, remote_offset: int, length: int):
        try:
            header = remote_offset.to_bytes(8, 'little') + length.to_bytes(8, 'little')
            sock.sendall(header)
            
            if not hasattr(self, 'memfd') or self.memfd is None:
                raise RuntimeError("memfd not registered")
            
            sent = 0
            while sent < length:
                n = os.sendfile(sock.fileno(), self.memfd, local_offset + sent, min(length - sent, 2147483647))
                if n == 0: raise RuntimeError(f"sendfile 0 at offset {local_offset+sent}/{self.buffer_length}")
                sent += n
            
            return True
        except Exception as e:
            logger.error(f"Send error to {target_host}:{target_port}: {e}")
            return False
        finally:
            sock.close()

    def transfer_submit_write(self, session_id: str, buffer: int, peer_buffer_address: int, length: int) -> int:
        parts = session_id.split(':')
        target_host = parts[0]
        target_ports = [int(p) for p in parts[1:]]
        
        if self.buffer_ptr is not None:
            local_offset = buffer - self.buffer_ptr
        else:
            local_offset = buffer
        
        overall_start_time = time.perf_counter()
        with self.batch_id_lock:
            batch_id = self.next_batch_id
            self.next_batch_id += 1
            self.pending_transfers[batch_id] = {'status': 0, 'length': length}
        
        chunk_size = length // self.num_parallel_streams
        futures = []
        
        for i in range(self.num_parallel_streams):
            chunk_offset = local_offset + i * chunk_size
            remote_chunk_offset = i * chunk_size
            chunk_length = length - (i * chunk_size) if i == self.num_parallel_streams - 1 else chunk_size
            
            # Fix IndexError: cycle through available ports if fewer than num_parallel_streams
            port = target_ports[i % len(target_ports)]
            sock = self._create_connection(target_host, port)
            futures.append(self.transfer_executor.submit(
                self._send_data_chunk, sock, target_host, port,
                chunk_offset, remote_chunk_offset, chunk_length
            ))
        
        def update_status():
            success = all(f.result() for f in futures)
            overall_duration = time.perf_counter() - overall_start_time
            if overall_duration > 0:
                aggregate_bw = (length / 1024 / 1024) / overall_duration
                logger.info(f"Aggregate Transfer: {length/1024/1024:.2f}MB in {overall_duration:.3f}s ({aggregate_bw:.2f}MB/s) using {self.num_parallel_streams} streams")
            
            with self.batch_id_lock:
                if batch_id in self.pending_transfers:
                    self.pending_transfers[batch_id]['status'] = 1 if success else -1
        
        self.transfer_executor.submit(update_status)
        return batch_id

    def transfer_check_status(self, batch_id: int) -> int:
        with self.batch_id_lock:
            return self.pending_transfers.get(batch_id, {}).get('status', -1)

    def get_session_id(self):
        if self.is_receiver and self.listener_ports:
            return f"{self.config.local_hostname}:{':'.join(str(p) for p in self.listener_ports)}"
        return f"{self.config.local_hostname}:{self.config.handshake_port}"

    def get_rpc_port(self):
        return self.listener_ports[0] if self.is_receiver and self.listener_ports else self.config.handshake_port
