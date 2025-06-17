import time
import zmq
from typing import List, Any
import os
import logging

from sglang.srt.coordinator.remote_manager import RolloutManager, RolloutConfig
from sglang.srt.utils import get_zmq_socket

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# standard format of sglang output
"""
{
    "index": int,
    "text": str,
    "meta_info": {
        "id": str,
        "finish_reason": dict,
        "prompt_tokens": int,
        "completion_tokens": int,
        "cached_tokens": int,
        "e2e_latency": float
    }
}
"""

class RolloutIterator:
    """
    Iterator class for the main process to communicate with the rollout manager.
    """
    def __init__(self, config: RolloutConfig, ipc_dir: str = "/tmp/rollout_ipc"):
        self.config = config
        self.ipc_dir = ipc_dir
        
        # Create IPC paths (same as in RolloutManager)
        self.prompt_ipc_path = f"ipc://{os.path.join(self.ipc_dir, 'rollout_prompt')}"
        self.result_ipc_path = f"ipc://{os.path.join(self.ipc_dir, 'rollout_result')}"
        self.control_ipc_path = f"ipc://{os.path.join(self.ipc_dir, 'rollout_control')}"
        
        # Initialize ZMQ context and sockets
        self.context = zmq.Context()
        
        # Socket to send prompts to the rollout manager
        self.prompt_socket = get_zmq_socket(
            self.context, zmq.PUSH, self.prompt_ipc_path, bind=False
        )
        
        # Socket to receive results from the rollout manager
        self.result_socket = get_zmq_socket(
            self.context, zmq.PULL, self.result_ipc_path, bind=False
        )
        
        # Socket for control commands
        self.control_socket = self.context.socket(zmq.REQ)  # Using regular socket for REQ type
        self.control_socket.connect(self.control_ipc_path)
        
        # State
        self.rollout_process = None
        self._start_time = None

        # local results for a unified interface
        self.local_results = None
        self.prompt_index_offset = 0

    def start_rollout_manager(self):
        """Start the rollout manager process"""
        manager = RolloutManager(self.config, self.ipc_dir)
        pid = manager.start()
        self.rollout_process = manager.process
        return pid
        
    def send_prompts(self, prompts: List[str]):
        """Send prompts to the rollout manager"""
        self.prompt_socket.send_pyobj(prompts)

    def add_local_results(self, results: List[Any]):
        """Add local results to the rollout iterator"""
        if self.local_results is None:
            # chunk the results into list of lists
            if len(results) % self.config.chunk_size != 0:
                logger.error(f"The number of results is not divisible by the chunk size: {len(results)} % {self.config.chunk_size} != 0.\nSkipping the local results...")
                return
            
            if self.config.out_of_order:
                # results should be a list of dicts
                if not isinstance(results[0], dict):
                    logger.error("Local results should be a list of dicts when out-of-order is True.\nSkipping the local results...")
                    return
                self._index_offset = len(results)
            else:
                # results should be a list of lists
                if not isinstance(results[0], list):
                    logger.error("Local results should be a list of lists when enforce in order.\nSkipping the local results...")
                    return
                self._index_offset = len(results) * self.config.n_samples

            logger.debug(f"Chunking {len(results)} local results into {len(results) // self.config.chunk_size} chunks")
            self.local_results = [results[i:i+self.config.chunk_size] for i in range(0, len(results), self.config.chunk_size)]
        else:
            logger.error("Local results can only be added once, multi-node is not supported yet.\nSkipping the local results...")
            return

    def get_status(self):
        """Get status information from the rollout manager"""
        self.control_socket.send_pyobj("status")
        return self.control_socket.recv_pyobj()
        
    def get_tracker(self):
        """Get tracker information from the rollout manager"""
        self.control_socket.send_pyobj("tracker")
        return self.control_socket.recv_pyobj()
        
    def shutdown(self):
        """Shutdown the rollout manager"""
        try:
            self.control_socket.send_pyobj("shutdown")
            return self.control_socket.recv_pyobj()
        except:
            # Force terminate if communication fails
            if self.rollout_process and self.rollout_process.is_alive():
                self.rollout_process.terminate()
                self.rollout_process.join(timeout=5)
                
    def flush(self):
        """Flush any remaining results"""
        self.control_socket.send_pyobj("flush")
        return self.control_socket.recv_pyobj()
        
    def __iter__(self):
        self._start_time = time.monotonic()
        return self
        
    def __next__(self):
        if self._start_time is None:
            self._start_time = time.monotonic()

        # if local results are not None, return them first
        if self.local_results:
            results = self.local_results.pop(0)
            logger.debug(f"Returning local chunk, chunks remaining: {len(self.local_results)}")
            return results, time.monotonic() - self._start_time
            
        # Poll for results with timeout
        if self.result_socket.poll(2000) == 0:
            # Check if we should continue waiting
            status = self.get_status()
            logger.info(f"Timeout status: \n{status}")

            # Check if all processing is done using the tracker
            tracker = status.get("tracker", {})
            if (tracker.get("all_samples_completed", False) and 
                tracker.get("all_samples_returned", False)):
                self._start_time = None
                raise StopIteration
                
            # Fallback to the old check method
            if (status["pending_prompts"] == 0 and 
                status["pending_tasks"] <= 2 and  # Only control and prompt handling tasks
                status["buffered_completed_requests"] == 0 and
                status["buffered_completed_prompts"] <= 0):
                self._start_time = None
                raise StopIteration
                
            return self.__next__()  # Try again
            
        # Get the next chunk
        chunk = self.result_socket.recv_pyobj()
        duration = time.monotonic() - self._start_time
        self._start_time = time.monotonic()  # Reset for next chunk
        
        if not chunk:
            logger.debug("Received empty chunk, stopping")
            self._start_time = None
            raise StopIteration

        # translate the index to the global index
        for result in chunk:
            if self.config.out_of_order:
                result["index"] += self.prompt_index_offset
            else:
                for sample in result:
                    sample["index"] += self.prompt_index_offset
        
        logger.debug(f"Received remote chunk, duration: {duration}")
        return chunk, duration
        
    def close(self):
        """Close the iterator and clean up resources"""
        self.flush()
        self.shutdown()
        
        # Close ZMQ sockets
        self.prompt_socket.close()
        self.result_socket.close()
        self.control_socket.close()
        self.context.term()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
