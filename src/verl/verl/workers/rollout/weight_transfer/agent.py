import logging
import os
import threading
import time
from typing import List, Tuple

import rpyc
import torch
import torch.multiprocessing as mp
import zmq

from .transfer_engine import MooncakeTransferEngine
from .utils import TransferAgentConfig, TransferStatus

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TransferRpycServer(rpyc.Service):
    def __init__(self, transfer_agent: "TransferAgent"):
        self.transfer_agent = transfer_agent

    def exposed_register_sglang_instance(
        self, 
        sglang_http_host: str,
        sglang_http_port: int, 
        mooncake_session_id: str, 
        buffer_ptr: int, 
        buffer_length: int,
        zmq_endpoint: str,
        zmq_port: int,
        # For P2P handshake
        mooncake_handshake_port: int = None
    ):
        try:
            # Create a unique identifier for this sglang instance
            instance_id = f"{sglang_http_host}:{sglang_http_port}"
            logger.info(f"[Weight sender] Registering sglang instance: {instance_id}, remote_session_id: {mooncake_session_id}")
            
            # Register with the transfer agent
            self.transfer_agent.register_receiver_session(
                instance_id, 
                mooncake_session_id,
                buffer_ptr,
                buffer_length,
                zmq_endpoint,
                zmq_port,
                sglang_http_host,
                sglang_http_port,
                mooncake_handshake_port
            )
            
            # Return the registration information
            return {
                "trainer_global_rank": self.transfer_agent.config.trainer_global_rank,
                "trainer_world_size": self.transfer_agent.config.trainer_world_size,
                "trainer_session_id": self.transfer_agent.get_session_id(),
                "trainer_buffer_ptr": self.transfer_agent.buffer.ptr,
                "trainer_buffer_length": self.transfer_agent.buffer.length,
                # For P2P handshake
                "trainer_hostname": self.transfer_agent.get_hostname(),
                "trainer_rpc_port": self.transfer_agent.get_rpc_port(),
            }
        
        except Exception as e:
            logger.error(f"Failed to register sglang instance: {e}")
            return {"status": "error", "message": str(e)}
    

class TransferBuffer:
    def __init__(self, params: List[Tuple[str, torch.Tensor]]):
        num_bytes = sum(p[1].numel() * p[1].element_size() for p in params)
        self.buffer = torch.zeros(num_bytes, dtype=torch.uint8, device='cpu')
        # NOTE(yongji): Now the buffer needs to be placed in share memory
        # Otherwise, the underlying storage will move when the transfer buffer is put into the mp.Queue,
        # and the ptr obtained in the next line will be invalid
        self.buffer.share_memory_()
        self.ptr = self.buffer.data_ptr()
        self.length = self.buffer.numel()


class TransferAgent:
    def __init__(
        self, 
        input_queue: mp.Queue, 
        output_queue: mp.Queue, 
        config: TransferAgentConfig,
    ):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.config = config
        self.registered_receivers = {}
        self.buffer = None
        self.mooncake_engine = None
        self.transfer_counter = 0
        self.endpoint = f"{os.environ.get('RAY_LOCAL_IP')}:{self.config.rpyc_bind_port}"
        self.initialize_mooncake_engine()
    
    def initialize_mooncake_engine(self):
        if isinstance(self.config.mooncake_config, str):
            # config is provided as a path
            self.mooncake_engine = MooncakeTransferEngine(
                config_path=self.config.mooncake_config
            )
        else:
            self.mooncake_engine = MooncakeTransferEngine(
                config=self.config.mooncake_config
            )
        self.session_id = self.mooncake_engine.get_session_id()

    def allocate_transfer_buffer(self, params: List[Tuple[str, torch.Tensor]]):
        assert self.mooncake_engine is not None, "Mooncake Engine not initialized"
        assert all(p[1].is_meta for p in params), "Meta tensors should be provided to compute buffer size"
        self.buffer = TransferBuffer(params)
        # put buffer in shared memory
        self.output_queue.put(self.buffer.buffer)
        self.mooncake_engine.register(self.buffer.ptr, self.buffer.length)
    
    def get_hostname(self):
        return self.mooncake_engine.get_hostname()
    
    def get_rpc_port(self):
        return self.mooncake_engine.get_rpc_port()

    def event_loop(self):
        try:
            while True:
                request = self.input_queue.get()
                if request == "wait_for_receivers":
                    instance_ids = self.input_queue.get()
                    logger.info(f"[Weight sender] Waiting for receivers: {instance_ids}")
                    self.wait_for_receiver_registration(instance_ids)
                    self.output_queue.put("completed")
                    continue
                    
                assert request == "send_weights"
                for instance_id in self.registered_receivers.keys():
                    self.transfer_weights_to_session(instance_id)
                self.output_queue.put("completed")
        except Exception as e:
            logger.error(f"Transfer agent event loop terminated: {e}")
            raise e

    def wait_for_receiver_registration(self, instance_ids: List[str]):
        while True:
            success = True
            for instance_id in instance_ids:
                if instance_id not in self.registered_receivers:
                    success = False
                    break
            if success:
                return
            time.sleep(1)

    def get_session_id(self):
        return self.mooncake_engine.get_session_id()
    
    def register_receiver_session(
        self, 
        instance_id, 
        session_id, 
        buffer_ptr, 
        buffer_length,
        zmq_endpoint, 
        zmq_port,
        sglang_http_host=None,
        sglang_http_port=None,
        mooncake_handshake_port=None
    ):
        if instance_id in self.registered_receivers:
            logger.warning(f"[Weight sender] Instance {instance_id} (session_id: {session_id}) already registered")
            
        assert self.buffer is not None, "Transfer buffer not allocated"
        # FIXME(yongji): Here we only consider full weight transfer
        assert self.buffer.length == buffer_length, \
            "Transfer buffer length mismatch between sender and receiver"

        self.registered_receivers[instance_id] = (
            session_id, buffer_ptr, buffer_length, 
            zmq_endpoint, zmq_port,
            sglang_http_host, sglang_http_port, mooncake_handshake_port
        )
            
        logger.info(f"[Weight sender] Registered rollout instance {instance_id} with weight receiver session ID {session_id}")
        return True

    def transfer_weights_to_session(self, instance_id):
        assert instance_id in self.registered_receivers, f"Instance {instance_id} not registered"
        target_session_id, remote_buffer_ptr, remote_buffer_length, _, _, hostname, _, handshake_port = \
            self.registered_receivers[instance_id]

        try:
            logger.info(f"[Weight sender] Transferring weights to session {target_session_id} (instance: {instance_id})")
            status = self.mooncake_engine.transfer_sync(
                target_session_id, 
                self.buffer.ptr, 
                remote_buffer_ptr, 
                remote_buffer_length
            )
            if status == 0:
                status = TransferStatus.SUCCESS
            else:
                status = TransferStatus.FAILURE
            self.sync_status_to_receiver_endpoint(instance_id, status)
            if status != TransferStatus.SUCCESS:
                raise Exception(f"Mooncake Transfer failed with status {status}")
        except Exception as e:
            logger.error(f"Failed to transfer weights: {e}")
            return {"status": "error", "message": str(e)}

    def sync_status_to_receiver_endpoint(self, instance_id, status):
        _, _, _, zmq_endpoint, zmq_port, _, _, _ = self.registered_receivers[instance_id]
        socket = zmq.Context().socket(zmq.PUSH)
        socket.connect(f"tcp://{zmq_endpoint}:{zmq_port}")
        socket.send_multipart(
            [
                str(self.config.trainer_global_rank).encode('ascii'),
                str(int(status)).encode('ascii'),
            ]
        )


def _init(
    config: TransferAgentConfig,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    event,
):
    transfer_agent = TransferAgent(input_queue, output_queue, config)

    from rpyc.utils.server import ThreadedServer

    # Create RPyC service with reference to this agent
    service = TransferRpycServer(transfer_agent)
    # Start server in a separate thread
    server = ThreadedServer(service, port=config.rpyc_bind_port, protocol_config={"allow_pickle": True})
    logger.info(f"Starting RPyC server on 0.0.0.0:{config.rpyc_bind_port}")
    threading.Thread(target=server.start, daemon=True).start()

    weights_meta_tensors = input_queue.get()
    # allocate_transfer_buffer will complete output_queue.put(weights_meta_tensors)
    transfer_agent.allocate_transfer_buffer(weights_meta_tensors)

    # FIXME(yongji):
    # Consider whether to register a graceful shutdown like LightLLM
    event.set()
    transfer_agent.event_loop()


def start_transfer_agent(
    config: TransferAgentConfig,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
):
    event = mp.Event()
    proc = mp.Process(target=_init, args=(config, input_queue, output_queue, event))
    proc.start()
    event.wait()
    assert proc.is_alive(), "Transfer agent process should be alive"
    logger.info("Successfully started sender transfer agent process.")
    return proc