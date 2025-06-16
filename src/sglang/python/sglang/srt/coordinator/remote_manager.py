import asyncio
import aiohttp
import json
import logging
import multiprocessing
import os
import pickle
import signal
import sys
import time
import threading
import zmq
import zmq.asyncio
from typing import List, Dict, Any, Optional, Tuple, Set, Deque, Counter
from collections import deque, defaultdict
from dataclasses import dataclass

from sglang.srt.coordinator.prompt_tracker import PromptTracker
from sglang.srt.utils import get_zmq_socket

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RolloutConfig:
    """Configuration for the rollout manager"""
    hosts: List[str] # ip:port
    max_tokens: int = 1000
    temperature: float = 0
    n_samples: int = 1
    return_tokens: bool = False
    out_of_order: bool = True
    chunk_size: int = 10

class RolloutManager:
    """
    Manages batched generation of responses from remote servers.
    Runs in a separate process and communicates with the main process via ZMQ.
    """
    def __init__(self, config: RolloutConfig, ipc_dir: str):
        self.config = config
        self.ipc_dir = ipc_dir
        
        # Create IPC paths
        # os.makedirs(self.ipc_dir, exist_ok=True)
        self.prompt_ipc_path = f"ipc://{os.path.join(self.ipc_dir, 'rollout_prompt')}"
        self.result_ipc_path = f"ipc://{os.path.join(self.ipc_dir, 'rollout_result')}"
        self.control_ipc_path = f"ipc://{os.path.join(self.ipc_dir, 'rollout_control')}"
        
        # ZMQ context and sockets will be initialized in the process
        self.context = None
        self.prompt_socket = None
        self.result_socket = None
        self.control_socket = None
        
        # Processing state
        self.pending_prompts = []
        self.is_running = False
        self.pending_tasks = set()
        self.completed_requests = []
        
        # For in-order mode
        self.prompt_buffer = {}  # request_idx -> list of results
        self.completed_prompts_buffer = []
        
        # Tracking data
        self.tracker = PromptTracker()
        
        # Process
        self.process = None
        self.loop = None

    def start(self):
        """Start the rollout manager process"""
        self.process = multiprocessing.Process(target=self._run_process)
        self.process.daemon = True
        self.process.start()
        logger.info(f"Started rollout manager process (PID: {self.process.pid})")
        return self.process.pid

    def _run_process(self):
        """Main function running in the subprocess"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize ZMQ
        self.context = zmq.asyncio.Context()
        
        # Socket to receive prompts from main process
        self.prompt_socket = get_zmq_socket(
            self.context, zmq.PULL, self.prompt_ipc_path, bind=True
        )
        
        # Socket to send results back to main process
        self.result_socket = get_zmq_socket(
            self.context, zmq.PUSH, self.result_ipc_path, bind=True
        )
        
        # Socket for control commands
        self.control_socket = self.context.socket(zmq.REP)  # Using regular socket for REP type
        self.control_socket.bind(self.control_ipc_path)
        
        # Run event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._main_loop())
        except Exception as e:
            logger.error(f"Error in rollout manager: {e}")
        finally:
            # Properly await the cleanup coroutine
            self.loop.run_until_complete(self._cleanup())

    async def _main_loop(self):
        """Main event loop for the rollout manager"""
        self.is_running = True
        
        # Create tasks for processing prompts and handling control commands
        prompt_task = self.loop.create_task(self._process_prompts())
        control_task = self.loop.create_task(self._handle_control_commands())
        
        self.pending_tasks.add(prompt_task)
        self.pending_tasks.add(control_task)
        
        # Wait for all tasks
        while self.is_running:
            done, pending = await asyncio.wait(
                self.pending_tasks, 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                self.pending_tasks.remove(task)
                
                if task.exception():
                    logger.error(f"Task error: {task.exception()}")
                    
                # Add new tasks from finished ones if needed
                if task is prompt_task and self.is_running:
                    prompt_task = self.loop.create_task(self._process_prompts())
                    self.pending_tasks.add(prompt_task)
                    
                if task is control_task and self.is_running:
                    control_task = self.loop.create_task(self._handle_control_commands())
                    self.pending_tasks.add(control_task)

    async def _process_prompts(self):
        """Process incoming prompts"""
        while self.is_running:
            try:
                # Get the next prompt
                prompt_data = await self.prompt_socket.recv_pyobj()
                logger.debug(f"Received prompt data: {type(prompt_data)}")
                
                if prompt_data is None:
                    logger.debug("Received None prompt, ignoring")
                    continue
                    
                # Handle batch of prompts
                if isinstance(prompt_data, list):
                    self.pending_prompts.extend(prompt_data)
                else:
                    self.pending_prompts.append(prompt_data)
                
                # Start processing if we have prompts
                if self.pending_prompts:
                    task = self.loop.create_task(self._process_batch(self.pending_prompts))
                    self.pending_tasks.add(task)
                    self.pending_prompts = []
                
            except asyncio.CancelledError:
                logger.debug("Prompt processing task was cancelled")
                break
            except Exception as e:
                logger.error(f"Error processing prompts: {e}")

    async def _process_batch(self, prompts: List[str]):
        """Process a batch of prompts by sending to servers and collecting results"""
        try:
            # Update tracker with new prompts
            self.tracker.add_prompts(prompts, self.config.n_samples)
            
            # Distribute prompts across available servers
            shard_size = (len(prompts) - 1) // len(self.config.hosts) + 1
            shards = [prompts[i*shard_size:(i+1)*shard_size] 
                    for i in range(len(self.config.hosts))]
            
            index_offset = 0
            async with aiohttp.ClientSession() as session:
                tasks = []
                for i, host in enumerate(self.config.hosts):
                    if i < len(shards) and shards[i]:
                        task = self.loop.create_task(
                            self._batch_generate_async(
                                session, shards[i], host, index_offset
                            )
                        )
                        tasks.append(task)
                        index_offset += len(shards[i]) * self.config.n_samples
                
                if tasks:
                    await asyncio.gather(*tasks)
            
            logger.info("Batch processing complete")
        except Exception as e:
            logger.error(f"Error during batch processing: {e}")
        finally:
            self.pending_tasks.remove(asyncio.current_task())

    async def _batch_generate_async(self, 
                                   session: aiohttp.ClientSession, 
                                   prompts: List[str],
                                   host: str, 
                                   index_offset: int) -> None:
        """Generate responses from a remote server for a batch of prompts"""
        url = f"http://{host}/generate_async"
        payload = {
            "text": prompts,
            "sampling_params": {
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_tokens,
                "n": self.config.n_samples
            },
            "stream": False,
            "return_logprob": self.config.return_tokens
        }

        try:
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                #TODO: make sure the response is blocked when the server is not ready (e.g. updating weights)
                async for line in response.content:
                    if line == b"\n":
                        continue
                    try:
                        data = json.loads(line.decode('utf-8').replace('data: ', ''))
                        if data == '[DONE]':
                            break
                        data["index"] += index_offset # NOTE: translate the index to the global index
                        await self._handle_response(data)
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding JSON: {line}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing response line: {e}")
        except aiohttp.ClientError as e:
            logger.error(f"HTTP Client Error for {host}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during request to {host}: {e}")

    async def _handle_response(self, data: Dict[str, Any]):
        """Handle a response from a server"""
        # Update tracker for completed sample
        self.tracker.mark_sample_completed(data["index"], self.config.n_samples)
        
        if self.config.out_of_order:
            # In out-of-order mode, send results as they come
            self.completed_requests.append(data)
            
            # Send chunk when we have enough
            if len(self.completed_requests) >= self.config.chunk_size or self.tracker.all_samples_completed:
                chunk = self.completed_requests[:self.config.chunk_size]
                await self._send_chunk(chunk)
                self.completed_requests = self.completed_requests[self.config.chunk_size:]
        else:
            # In in-order mode, group by original prompt
            global_idx = data["index"]
            request_idx = global_idx // self.config.n_samples
            
            if request_idx not in self.prompt_buffer:
                self.prompt_buffer[request_idx] = []
            
            self.prompt_buffer[request_idx].append(data)
            
            # If we've received all samples for this prompt, move it to completed
            if len(self.prompt_buffer[request_idx]) == self.config.n_samples:
                completed_samples = self.prompt_buffer.pop(request_idx)
                self.completed_prompts_buffer.append(completed_samples)
                
                # Send a chunk when we have enough
                if len(self.completed_prompts_buffer) >= self.config.chunk_size or self.tracker.all_samples_completed:
                    chunk = self.completed_prompts_buffer[:self.config.chunk_size]
                    await self._send_chunk(chunk)
                    self.completed_prompts_buffer = self.completed_prompts_buffer[self.config.chunk_size:]

    async def _send_chunk(self, chunk):
        """Send a chunk of results back to the main process"""
        # Update tracker for returned samples
        self.tracker.mark_samples_returned(chunk, self.config.n_samples)
        
        await self.result_socket.send_pyobj(chunk)
        logger.debug(f"Sent chunk of size {len(chunk)}")

    async def _handle_control_commands(self):
        """Handle control commands from the main process"""
        while self.is_running:
            try:
                command = await self.control_socket.recv_pyobj()
                logger.debug(f"Received control command: {command}")
                
                if command == "status":
                    # Return status information
                    if self.config.out_of_order:
                        status = {
                            "pending_prompts": len(self.pending_prompts),
                            "pending_tasks": len(self.pending_tasks),
                            "buffered_completed_prompts": -1, # ooo mode does not buffer completed prompts
                            "buffered_completed_requests": len(self.completed_requests),
                            "is_running": self.is_running,
                            "tracker": self.tracker.get_summary()
                        }
                    else:
                        buffered_requests = sum(len(requests) for requests in self.prompt_buffer.values())
                        status = {
                            "pending_prompts": len(self.pending_prompts),
                            "pending_tasks": len(self.pending_tasks),
                            "buffered_completed_prompts": len(self.completed_prompts_buffer),
                            "buffered_completed_requests": buffered_requests,
                            "is_running": self.is_running,
                            "tracker": self.tracker.get_summary()
                        }
                    await self.control_socket.send_pyobj(status)
                
                elif command == "tracker":
                    # Just return the tracker summary
                    await self.control_socket.send_pyobj(self.tracker.get_summary())
                
                elif command == "flush":
                    # Flush any remaining results
                    if self.config.out_of_order and self.completed_requests:
                        await self._send_chunk(self.completed_requests)
                        self.completed_requests = []
                    
                    elif not self.config.out_of_order and self.completed_prompts_buffer:
                        chunk = self.completed_prompts_buffer
                        self.completed_prompts_buffer = []
                        await self._send_chunk(chunk)
                    
                    await self.control_socket.send_pyobj({"status": "flushed"})
                
                elif command == "shutdown":
                    logger.info("Shutdown command received")
                    self.is_running = False
                    await self.control_socket.send_pyobj({"status": "shutting_down"})
                    break
                
                else:
                    await self.control_socket.send_pyobj({"status": "unknown_command"})
                    
            except asyncio.CancelledError:
                logger.debug("Control command task was cancelled")
                break
            except Exception as e:
                logger.error(f"Error handling control command: {e}")
                try:
                    await self.control_socket.send_pyobj({"status": "error", "message": str(e)})
                except:
                    pass

    def _signal_handler(self, signum, frame):
        """Handle signals to gracefully shut down"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
        if self.loop:
            self.loop.call_soon_threadsafe(self._initiate_shutdown)

    def _initiate_shutdown(self):
        """Initiate shutdown sequence"""
        for task in self.pending_tasks:
            task.cancel()
        
        self.loop.create_task(self._cleanup())

    async def _cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources...")
        
        # Close ZMQ sockets
        if self.prompt_socket:
            self.prompt_socket.close(linger=0)
        
        if self.result_socket:
            self.result_socket.close(linger=0)
        
        if self.control_socket:
            self.control_socket.close(linger=0)
        
        if self.context:
            self.context.term()
        
        # Stop the loop
        if self.loop and self.loop.is_running():
            self.loop.stop()
        
        logger.info("Cleanup complete")
