import logging
import socket
import threading
import time
import queue
import functools
from typing import List, Tuple, Dict, Optional

import torch
import torch.multiprocessing as mp
import rpyc
import zmq
from rpyc.utils.classic import obtain

from .transfer_engine import MooncakeTransferEngine
from .utils import TransferAgentConfig, TransferStatus
from ..utils import configure_logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TransferBuffer:
    def __init__(self, params: List[Tuple[str, torch.Tensor]]):
        num_bytes = sum(p[1].numel() * p[1].element_size() for p in params)
        # Initialize with zeros, actual weights will be transferred here
        self.buffer = torch.zeros(num_bytes, dtype=torch.uint8, device='cpu')
        # NOTE(yongji): Now we need to put the buffer in share memory
        # Otherwise, when the transfer buffer is put into the mp.Queue,
        # the underlying storage will move, causing the ptr obtained in the next line to be invalid
        self.buffer.share_memory_()
        self.ptr = self.buffer.data_ptr()
        self.length = self.buffer.numel()
        # self.internal_buffer = np.zeros(num_bytes, dtype=np.uint8)
        # self.ptr = self.internal_buffer.ctypes.data
        # self.length = self.internal_buffer.size


def retry(max_attempts=3, wait_time=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    logger.info(f"try {func.__name__} {attempts}/{max_attempts} fail: {str(e)}")
                    if attempts < max_attempts:
                        time.sleep(wait_time)
            raise Exception(f"{func.__name__} try all failed")

        return wrapper

    return decorator


class TransferAgent:
    def __init__(self, input_queue: mp.Queue, output_queue: mp.Queue, config: TransferAgentConfig):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = config
        self.buffer: Optional[TransferBuffer] = None
        self.mooncake_engine: Optional[MooncakeTransferEngine] = None
        self.zmq_listener_port: Optional[int] = None
        self.zmq_thread: Optional[threading.Thread] = None
        # sender rank -> info
        # FIXME(yongji): Now we only consider full weight sending
        # single sender to single/multiple receiver
        self.sender_info: Dict[int, Dict] = {}
        # sender rank -> status
        self._transfer_status_queue: queue.Queue[Tuple[int, TransferStatus]] = queue.Queue()

        self.initialize_mooncake_engine()
        self.start_zmq_server()

    def initialize_mooncake_engine(self):
        logger.info("Initializing Mooncake Transfer Engine...")
        if isinstance(self.config.mooncake_config, str):
            # config is provided as a path
            self.mooncake_engine = MooncakeTransferEngine(
                config_path=self.config.mooncake_config
            )
        else:
            # config is provided as an object
            self.mooncake_engine = MooncakeTransferEngine(
                config=self.config.mooncake_config
            )
        logger.info(f"Mooncake Engine initialized. Session ID: {self.get_session_id()}")

    def find_free_port(self) -> int:
        """Find an available free port on the host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def _zmq_listener_thread(self):
        """Thread function to listen for incoming status messages from senders."""
        context = zmq.Context()
        socket = context.socket(zmq.PULL)
        port = self.find_free_port()
        bind_addr = f"tcp://{self.config.zmq_bind_host}:{port}"
        self.zmq_port = port
        
        # Add retry logic for binding
        max_attempts = 5
        current_attempt = 0
        while current_attempt < max_attempts:
            try:
                socket.bind(bind_addr)
                logger.info(f"Successfully bound ZMQ socket to {bind_addr}")
                break
            except zmq.error.ZMQError as e:
                current_attempt += 1
                if current_attempt < max_attempts:
                    logger.warning(f"Failed to bind to {bind_addr}: {e}. Retry {current_attempt}/{max_attempts}...")
                    # Find a new port and try again
                    port = self.find_free_port()
                    bind_addr = f"tcp://{self.config.zmq_bind_host}:{port}"
                    self.zmq_port = port
                else:
                    logger.error(f"Failed to bind ZMQ socket after {max_attempts} attempts")
                    raise

        while True:
            try:
                # Use blocking recv_multipart directly
                sender_rank_bytes, status_bytes = socket.recv_multipart()
                sender_rank = int(sender_rank_bytes.decode('ascii'))
                status_code = int(status_bytes.decode('ascii'))
                status = TransferStatus(status_code)
                self._transfer_status_queue.put((sender_rank, status))
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    logger.info("[Weight receiver] ZMQ context terminated, exiting listener thread.")
                    break # Context terminated
                else:
                    logger.error(f"[Weight receiver] ZMQ Error in listener thread: {e}")
                    break
            except Exception as e:
                logger.error(f"[Weight receiver] Error in ZMQ listener thread: {e}", exc_info=True)
                raise e

    def start_zmq_server(self):
        """Starts the ZMQ listener thread."""
        if self.zmq_thread is None or not self.zmq_thread.is_alive():
            self.zmq_thread = threading.Thread(target=self._zmq_listener_thread, daemon=True)
            self.zmq_thread.start()
            # Wait a moment for the thread to start and bind the port
            time.sleep(0.5)
            while self.zmq_port is None and self.zmq_thread.is_alive():
                time.sleep(0.1)
            if self.zmq_port is None:
                 raise RuntimeError("Failed to start ZMQ server and bind port.")
            logger.info(f"[Weight receiver] ZMQ listener thread started successfully on port {self.zmq_port}.")
        else:
            logger.warning("[Weight receiver] ZMQ server thread is already running.")

    def allocate_transfer_buffer(self, params: List[Tuple[str, torch.Tensor]]):
        assert self.mooncake_engine is not None, "Mooncake Engine not initialized"
        assert all(p[1].is_meta for p in params), "Meta tensors should be provided to compute buffer size"
        self.buffer = TransferBuffer(params)
        self.output_queue.put(self.buffer.buffer)
        self.mooncake_engine.register(self.buffer.ptr, self.buffer.length)

    def get_local_ip(self) -> str:
        """Gets the local IP address used for outgoing connections."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('8.8.8.8', 1))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip

    def register_with_sender(self):
        assert self.mooncake_engine is not None, "Mooncake Engine not initialized"
        assert self.buffer is not None, "Transfer buffer not allocated"
        assert self.zmq_port is not None, "ZMQ server not started or port not available"

        for rpyc_host, rpyc_port in self.config.sender_rpyc_endpoints:
            sender_address = rpyc_host
            sender_port = rpyc_port
            logger.info(f"Attempting to connect to sender RPyC server at {sender_address}:{sender_port}...")

            try:
                if self.config.zmq_bind_host != "0.0.0.0":
                    zmq_endpoint = self.config.zmq_bind_host
                else:
                    # Use the new method to get the local IP
                    zmq_endpoint = self.get_local_ip()
                    logger.info(f"[Weight receiver] Determined local IP for ZMQ endpoint: {zmq_endpoint}")

                conn = retry(max_attempts=5)(rpyc.connect)(sender_address, sender_port, config={"allow_pickle": True})
                logger.info(f"[Weight receiver] Connected to sender {sender_address}:{sender_port} RPyC server.")

                registration_info = {
                    "sglang_http_host": self.config.sglang_http_host,
                    "sglang_http_port": self.config.sglang_http_port,
                    "mooncake_session_id": self.get_session_id(),
                    "buffer_ptr": self.buffer.ptr,
                    "buffer_length": self.buffer.length,
                    "zmq_endpoint": zmq_endpoint, # Hostname/IP sender should connect to
                    "zmq_port": self.zmq_port, # Port sender should connect to
                    "mooncake_handshake_port": self.mooncake_engine.get_rpc_port() # Port for P2P handshake
                }
                logger.info(f"[Weight receiver] Registering with sender: {registration_info}")

                # Call the exposed method on the sender
                response = conn.root.register_sglang_instance(**registration_info)
                response = obtain(response)
                conn.close()

                if isinstance(response, dict) and response.get("trainer_session_id"):
                    sender_rank = response["trainer_global_rank"]
                    self.sender_info[sender_rank] = response
                    logger.info(f"[Weight receiver] Successfully registered with sender. Received info: {self.sender_info}")
                    return True
                else:
                    logger.error(f"[Weight receiver] Registration failed. Sender response: {response}")
                    return False

            except ConnectionRefusedError:
                logger.error(f"[Weight receiver] Connection refused when trying to connect to RPyC server at {sender_address}:{sender_port}. Is the sender agent running?")
                return False
            except Exception as e:
                logger.error(f"[Weight receiver] Failed to register with sender RPyC server: {e}", exc_info=True)
                return False

    def wait_for_transfer_completion(self) -> TransferStatus:
        remaining_senders = set(self.sender_info.keys())
        while len(remaining_senders) > 0:
            sender_rank, status = self._transfer_status_queue.get()
            if status == TransferStatus.SUCCESS:
                remaining_senders.remove(sender_rank)
            elif status == TransferStatus.FAILURE:
                return TransferStatus.FAILURE
        # np.copyto(
        #     self.buffer.buffer.numpy(force=False), 
        #     self.buffer.internal_buffer, 
        #     casting='no'
        # )
        return TransferStatus.SUCCESS
    
    def event_loop(self):
        try:
            while True:
                request = self.input_queue.get()
                assert request == "receive_weights"
                status = self.wait_for_transfer_completion()
                if status != TransferStatus.SUCCESS:
                    raise RuntimeError(f"Weight transfer failed")
                self.output_queue.put("completed")
        except BaseException as e:
            logger.error(f"Transfer agent event loop terminated: {e}")
            raise e

    def get_session_id(self) -> str:
        assert self.mooncake_engine is not None, "Mooncake Engine not initialized"
        return self.mooncake_engine.get_session_id()


def _init(
    config: TransferAgentConfig,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    event,
    server_args,
):
    # Configure logging for the subprocess using sglang's configure_logger
    configure_logger(server_args, prefix=" Weight Transfer Agent")
    transfer_agent = TransferAgent(input_queue, output_queue, config)

    weights_meta_tensors = input_queue.get()
    transfer_agent.allocate_transfer_buffer(weights_meta_tensors)
    logger.info("Transfer buffer allocated and placed in output queue.")
    if not transfer_agent.register_with_sender():
         raise RuntimeError("Failed to register with sender RPyC server.")

    event.set()
    transfer_agent.event_loop() # This runs indefinitely


def start_transfer_agent(
    config: TransferAgentConfig,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    server_args=None,
):
    event = mp.Event()
    proc = mp.Process(target=_init, args=(config, input_queue, output_queue, event, server_args), daemon=True)
    proc.start()
    event.wait() # Wait until the _init function signals readiness
    assert proc.is_alive(), "Transfer agent process should be alive"
    logger.info("Successfully started receiver transfer agent process.")
    return proc

