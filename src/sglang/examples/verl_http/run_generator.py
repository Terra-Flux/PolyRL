import asyncio
import aiohttp
import json
import time
from typing import List, Any, AsyncIterator, Dict, Tuple
import queue
from threading import Thread
from typing import Iterator, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AsyncBatchGenerator:
    def __init__(self, prompts: List[str], hosts: List[str], ports: List[int], 
                 max_tokens: int = 1000, temperature: float = 0, n_samples: int = 1, loop=None):
        self.prompts = prompts
        self.hosts = hosts
        self.ports = ports
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n = n_samples
        self.queue = asyncio.Queue()
        self.is_running = False
        self.generation_task = None
        self.loop = loop

    async def _batch_generate_async(self, session: aiohttp.ClientSession, prompts: List[str], 
                                  host: str, port: int, index_offset: int) -> None:
        url = f"http://{host}:{port}/generate_async"
        payload = {
            "text": prompts,
            "sampling_params": {
                "temperature": self.temperature,
                "max_new_tokens": self.max_tokens,
                "n": self.n
            },
            "stream": False
        }

        try:
            async with session.post(url, json=payload) as response:
                 response.raise_for_status()
                 async for line in response.content:
                    if line == b"\n":
                        continue
                    try:
                        data = json.loads(line.decode('utf-8').replace('data: ', ''))
                        if data == '[DONE]':
                            break
                        data["index"] += index_offset
                        await self.queue.put(data)
                    except json.JSONDecodeError:
                        logger.error(f"Error decoding JSON: {line}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing response line: {e}")
                        await self.queue.put(e)
        except aiohttp.ClientError as e:
             logger.error(f"HTTP Client Error for {host}:{port}: {e}")
             await self.queue.put(e)
        except Exception as e:
            logger.error(f"Unexpected error during request to {host}:{port}: {e}")
            await self.queue.put(e)


    async def _run_generation(self):
        try:
            shard_size = (len(self.prompts) - 1) // len(self.hosts) + 1
            shards = [self.prompts[i*shard_size:(i+1)*shard_size] 
                     for i in range(len(self.hosts))]

            index_offset = 0
            async with aiohttp.ClientSession() as session:
                tasks = []
                for i, (host, port) in enumerate(zip(self.hosts, self.ports)):
                    if i < len(shards) and shards[i]:
                        task = asyncio.create_task(
                            self._batch_generate_async(session, shards[i], host, port, index_offset)
                        )
                        tasks.append(task)
                        index_offset += len(shards[i]) * self.n
                
                if tasks:
                    await asyncio.gather(*tasks)
            
            logger.info("===========Async Generation complete===========")
        except Exception as e:
            logger.error(f"Error during async generation: {e}")
            await self.queue.put(e)
        finally:
            self.is_running = False
            await self.queue.put(None)

    async def __aiter__(self) -> AsyncIterator[Union[Any, Exception, None]]:
        if not self.is_running:
            self.is_running = True
            current_loop = self.loop or asyncio.get_event_loop()
            self.generation_task = current_loop.create_task(self._run_generation())

        try:
            while True:
                item = await self.queue.get()
                if item is None:
                    logger.info("===========Async Iteration complete===========")
                    yield None
                    break 
                yield item
        except asyncio.CancelledError:
            logger.warning("Async Generator Cancelled")
            if self.generation_task and not self.generation_task.done():
                self.generation_task.cancel()
            raise
        finally:
            if self.generation_task and not self.generation_task.done():
                self.generation_task.cancel()


    async def aclose(self):
        logger.debug("Closing AsyncBatchGenerator...")
        if self.generation_task and not self.generation_task.done(): # Restore the done() check
            logger.debug("Cancelling generation task...")
            self.generation_task.cancel()
            try:
                await self.generation_task
                logger.debug("Generation task cancelled.")
            except asyncio.CancelledError:
                logger.debug("Generation task cancellation confirmed.")
            except Exception as e:
                logger.error(f"Error during generation task cancellation: {e}")
        logger.debug("AsyncBatchGenerator closed.")


class SyncIteratorWrapper:
    def __init__(self, prompts: List[str], hosts: List[str], ports: List[int], 
                 chunk_size: int, 
                 out_of_order: bool = True,
                 max_tokens: int = 1000, temperature: float = 0, 
                 n_samples: int = 1):
        
        if not out_of_order and n_samples <= 0:
            raise ValueError("n_samples must be > 0 for in-order mode")

        self.sync_queue = queue.Queue()
        self.prompts = prompts
        self.chunk_size = chunk_size
        self.out_of_order = out_of_order
        self.n_samples = n_samples
        
        self._prompt_buffer: Dict[int, List[Any]] = {}
        self._completed_prompts_buffer: List[List[Any]] = []

        self.thread = Thread(target=self._run_async_loop, args=(
            prompts, hosts, ports, max_tokens, temperature, n_samples
        ))
        self.thread.daemon = False 
        self.loop = None
        self._stop_event = asyncio.Event()
        self._start_time = None
        self.thread.start()

    def _run_async_loop(self, prompts, hosts, ports, max_tokens, temperature, n_samples):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        async def async_generator_wrapper():
            generator = AsyncBatchGenerator(
                prompts=prompts, hosts=hosts, ports=ports, 
                max_tokens=max_tokens, temperature=temperature, 
                n_samples=n_samples, loop=self.loop
            )
            
            try:
                async for item in generator:
                    if self._stop_event.is_set():
                        logger.debug("Stop event set, breaking async loop.")
                        break
                    self.sync_queue.put(item)
                    if item is None:
                        logger.debug("Received None signal from async generator.")
                        break 
            except Exception as e:
                logger.error(f"Exception in async generator wrapper: {e}")
                if not self._stop_event.is_set():
                    self.sync_queue.put(e)
            finally:
                if not self._stop_event.is_set():
                    logger.debug("Putting final None on queue.")
                    self.sync_queue.put(None)
                await generator.aclose()

        try:
            self.loop.run_until_complete(async_generator_wrapper())
        finally:
            try:
                # Give pending tasks (like cancellation cleanup) a chance to complete
                pending = asyncio.all_tasks(loop=self.loop)
                # Filter out the current task wrapper if it's somehow still pending
                current_task = asyncio.current_task(loop=self.loop)
                pending = {task for task in pending if task is not current_task}
                if pending:
                    logger.debug(f"Running {len(pending)} pending tasks until completion before closing loop.")
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception as e:
                logger.error(f"Error during pending task cleanup before loop close: {e}")
            finally:
                if not self.loop.is_closed():
                    logger.debug("Closing event loop now.")
                    self.loop.close()
            logger.debug("Async loop finished.")

    def __iter__(self) -> Iterator[Tuple[Union[List[Any], List[List[Any]]], float]]:
        self._start_time = time.monotonic()
        return self

    # Helper method to get items from the queue, handling None and Exceptions
    def _get_next_item_from_queue(self) -> Any:
        item = self.sync_queue.get()
        if item is None:
            self.thread.join() # Ensure thread finishes before stopping iteration
            self._start_time = None # Reset timer for next iteration if any
            raise StopIteration
        if isinstance(item, Exception):
            self._start_time = None # Reset timer
            raise item
        return item

    def __next__(self) -> Tuple[Union[List[Any], List[List[Any]]], float]:
        if self._start_time is None:
             self._start_time = time.monotonic()

        start_collect_time = time.monotonic() # Track time for this batch
        try:
            if self.out_of_order:
                batch = self._next_ooo()
            else:
                batch = self._next_in_order()

            duration = time.monotonic() - self._start_time
            self._start_time = time.monotonic() # Reset start time for the *next* batch

            # Handle case where StopIteration was raised inside _next methods after collecting some items
            if not batch and self.sync_queue.empty():
                # This check might be redundant if StopIteration is handled correctly within _get_next_item_from_queue
                # but kept for safety for now.
                pass # Let StopIteration propagate if raised by _get_next_item_from_queue

            return batch, duration

        except StopIteration:
            # If StopIteration occurs while collecting the first item of a batch
            duration = time.monotonic() - start_collect_time # Duration of the failed attempt
            self._start_time = None # Ensure timer is reset
            logger.debug(f"StopIteration caught in __next__, duration: {duration:.2f}s")
            raise # Re-raise StopIteration


    def _next_ooo(self) -> List[Any]:
        chunk = []
        try:
            while len(chunk) < self.chunk_size:
                item = self._get_next_item_from_queue() # Use helper
                chunk.append(item)
        except StopIteration:
            # Expected when the queue is exhausted
            if not chunk:
                 raise # Re-raise if no items were collected for this chunk
            # Otherwise, return the partially filled chunk
        return chunk

    def _next_in_order(self) -> List[List[Any]]:
        current_chunk_prompts: List[List[Any]] = []

        # First, drain any fully completed prompts buffered from the previous call
        while self._completed_prompts_buffer and len(current_chunk_prompts) < self.chunk_size:
            current_chunk_prompts.append(self._completed_prompts_buffer.pop(0))

        # Then, collect new items from the queue until the chunk is full
        try:
            while len(current_chunk_prompts) < self.chunk_size:
                item = self._get_next_item_from_queue() # Use helper

                global_idx = item["index"]
                # Ensure n_samples is positive to avoid division by zero
                if self.n_samples <= 0:
                    logger.error(f"Invalid n_samples value ({self.n_samples}) encountered in _next_in_order.")
                    # Decide how to handle: skip item, raise error, etc.
                    # For now, let's skip this item as request_idx calculation is invalid.
                    continue 
                request_idx = global_idx // self.n_samples

                if request_idx < 0:
                    logger.warning(f"Warning: Invalid request_idx ({request_idx}) derived from global_idx {global_idx} and n_samples {self.n_samples}")
                    continue # Skip this potentially corrupt item

                if request_idx not in self._prompt_buffer:
                    self._prompt_buffer[request_idx] = []

                self._prompt_buffer[request_idx].append(item)

                # Check if all samples for this prompt are now received
                if len(self._prompt_buffer[request_idx]) == self.n_samples:
                    completed_samples = self._prompt_buffer.pop(request_idx)
                    # Add to current chunk or buffer for the next chunk
                    if len(current_chunk_prompts) < self.chunk_size:
                        current_chunk_prompts.append(completed_samples)
                    else:
                        self._completed_prompts_buffer.append(completed_samples)

        except StopIteration:
            # Queue exhausted. Add any remaining fully buffered prompts to the current chunk.
            if self._completed_prompts_buffer:
                remaining_needed = self.chunk_size - len(current_chunk_prompts)
                transfer_count = min(remaining_needed, len(self._completed_prompts_buffer))
                current_chunk_prompts.extend(self._completed_prompts_buffer[:transfer_count])
                self._completed_prompts_buffer = self._completed_prompts_buffer[transfer_count:]
            
            # If after all that, the chunk is still empty, re-raise StopIteration
            if not current_chunk_prompts:
                raise

        return current_chunk_prompts


    def close(self):
        logger.debug("Closing SyncIteratorWrapper...")
        if self.loop and not self.loop.is_closed():
            logger.debug("Signalling async loop to stop...")
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self._stop_event.set)
            else:
                self._stop_event.set()
            self.sync_queue.put(None) 
        else:
            self.sync_queue.put(None)

        if self.thread.is_alive():
            logger.debug("Joining async thread...")
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                logger.warning("Warning: Async thread did not exit cleanly.")
            else:
                logger.debug("Async thread joined.")
        logger.debug("SyncIteratorWrapper closed.")


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Example usage with proper cleanup
def main():
    prompts = [
        "<|im_start|>user\nHow do quantum computers differ from classical computers?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain the fermentation process in making sourdough bread<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat are the ethical implications of autonomous vehicles?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDescribe the cultural significance of tea ceremonies in Japan<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nHow does neuroplasticity affect learning in adults?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a short story about a time traveler who can only go forward in time<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat causes the aurora borealis phenomenon?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain how cryptocurrency mining impacts the environment<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nCompare and contrast impressionism and expressionism in art<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat are the most promising technologies for carbon capture?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDescribe the process of photosynthesis in desert plants<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nHow has the concept of privacy evolved in the digital age?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain the mechanics of black holes in simple terms<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat are the cultural differences in breakfast foods around the world?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nHow do bees communicate with each other?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat makes a programming language Turing complete?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDescribe the process of making traditional kimchi<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat are the psychological effects of color in marketing?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain how a violin produces sound<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the significance of dreams in different cultures?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nHow do coral reefs form and why are they important?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat are the origins of chess and how has it evolved?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain the concept of zero-knowledge proofs in cryptography<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nHow do migratory birds navigate across continents?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat are the differences between various coffee brewing methods?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDescribe the process of glass blowing and its history<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nHow does the human memory system work?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat are the philosophical implications of artificial intelligence?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain how vaccines train the immune system<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat causes the northern lights and where can they be seen?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nHow do different writing systems around the world represent language?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the relationship between mathematics and music?<|im_end|>\n<|im_start|>assistant\n", 
    ]

    n_samples = 10
    ooo_chunk_size = 10
    in_order_chunk_size = 2

    logger.info("--- Running Out-of-Order (OOO) Mode ---")
    logger.info(f"Chunk size: {ooo_chunk_size} (individual responses)")
    
    ooo_results = []
    total_ooo_results = 0
    total_ooo_time = 0
    with SyncIteratorWrapper(
        prompts=prompts, hosts=["localhost", "localhost"], ports=[30000, 30001],
        chunk_size=ooo_chunk_size, out_of_order=True,
        temperature=0.7, n_samples=n_samples
    ) as iterator_ooo:
        try:
            for i, (batch, duration) in enumerate(iterator_ooo):
                logger.info(f"\nOOO Batch {i+1} (size {len(batch)}) took {duration:.2f} seconds:")
                total_ooo_results += len(batch)
                total_ooo_time += duration
                batch_results = []
                for result in batch:
                    global_idx = result["index"]
                    request_idx = global_idx // n_samples
                    sample_idx = global_idx % n_samples
                    logger.info(f"  Got response for Prompt {request_idx}, Sample {sample_idx}")
                    batch_results.append({
                        "request_idx": request_idx,
                        "sample_idx": sample_idx,
                        "prompt": prompts[request_idx],
                        "result": result["output"][-1]["text"]
                    })
                ooo_results.extend(batch_results)
        except Exception as e:
            logger.error(f"Error during OOO iteration: {e}")
    logger.info(f"\nTotal OOO results received: {total_ooo_results} in {total_ooo_time:.2f} seconds")

    # Save OOO results to a JSON file
    with open("ooo_results.json", "w") as f:
        json.dump(ooo_results, f, indent=2)
    logger.info("OOO results saved to ooo_results.json")

    logger.info("--- Running In-Order Mode ---")
    logger.info(f"Chunk size: {in_order_chunk_size} (fully sampled prompts)")
    logger.info(f"Samples per prompt (n): {n_samples}")
    
    in_order_results = []
    total_in_order_prompts = 0
    total_in_order_samples = 0
    total_in_order_time = 0
    with SyncIteratorWrapper(
        prompts=prompts, hosts=["localhost", "localhost"], ports=[30000, 30001],
        chunk_size=in_order_chunk_size, out_of_order=False,
        temperature=0.7, n_samples=n_samples
    ) as iterator_in_order:
        try:
            for i, (batch, duration) in enumerate(iterator_in_order):
                logger.info(f"\nIn-Order Batch {i+1} (contains {len(batch)} fully sampled prompts) took {duration:.2f} seconds:")
                total_in_order_prompts += len(batch)
                total_in_order_time += duration
                for prompt_samples in batch:
                    total_in_order_samples += len(prompt_samples)
                    if not prompt_samples: continue
                    prompt_idx = prompt_samples[0]["index"] // n_samples
                    logger.info(f"  Got all {len(prompt_samples)} samples for Prompt {prompt_idx}:")
                    batch_samples = [result["output"][-1]["text"] for result in prompt_samples]
                    in_order_results.append({
                        "prompt_idx": prompt_idx,
                        "prompt": prompts[prompt_idx],
                        "samples": batch_samples
                    })

        except Exception as e:
            logger.error(f"Error during In-Order iteration: {e}")
            
    logger.info(f"\nTotal In-Order prompts completed: {total_in_order_prompts}")
    logger.info(f"Total In-Order samples received: {total_in_order_samples} in {total_in_order_time:.2f} seconds")

    # Save In-Order results to a JSON file
    with open("in_order_results.json", "w") as f:
        json.dump(in_order_results, f, indent=2)
    logger.info("In-Order results saved to in_order_results.json")

if __name__ == "__main__":
    main()