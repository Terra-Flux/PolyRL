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
from collections.abc import AsyncGenerator
import aiohttp
import json

class StreamingBatchIterator:
    """
    A self-contained synchronous iterator that fetches and batches items
    from a streaming HTTP endpoint based on a timeout.
    """
    def __init__(self, url: str, payload: dict, drain_timeout: float = 0.01, session_timeout: float = 3000):
        self.url = url
        self.drain_timeout = drain_timeout
        self.session_timeout = session_timeout
        self._loop = asyncio.get_event_loop()
        self._iterator = self._async_generator(payload)

    async def _async_generator(self, payload) -> AsyncGenerator[list, None]:
        """A private async generator that handles fetching and batching."""
        try:
            timeout = aiohttp.ClientTimeout(total=self.session_timeout)  # 3000 seconds
            # NOTE(liuxs): chunk size at least 1MB to avoid chunk too big error
            async with aiohttp.ClientSession(timeout=timeout, read_bufsize=2**24) as session:
                async with session.post(self.url, json=payload) as response:
                    response.raise_for_status() # Raise an exception for bad status codes

                    stream = response.content.__aiter__()
                    notifier = await stream.__anext__()
                    # NOTE(liuxs): yield the notifier so that the local context will switch
                    yield json.loads(notifier)

                    current_batch = []
                    while True:
                        fetch_task = asyncio.create_task(stream.__anext__())
                        done, _ = await asyncio.wait({fetch_task}, timeout=self.drain_timeout)

                        if done:
                            current_batch.append(fetch_task.result())
                        else: # Timeout
                            yield [json.loads(line) for line in current_batch if line]
                            current_batch = [] # Reset batch to prevent double-yield
                            # Now wait for the pending item
                            line = await fetch_task
                            if line:
                                current_batch = [line]
        except StopAsyncIteration:
            if current_batch:
                yield [json.loads(line) for line in current_batch if line]
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            print("Stream finished.")

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._loop.run_until_complete(self._iterator.__anext__())
        except StopAsyncIteration:
            self.close()
            raise StopIteration

    def close(self):
        """Closes the event loop."""
        # if not self._loop.is_closed():
        #     self._loop.close()
        #     print("Event loop closed.")
        pass
            
# example usage:
"""
def main():
    iterator = StreamingBatchIterator(
        url="http://localhost:10000/stream", 
        timeout=0.1
    )
    # get status first to make sure request is submitted successfully
    status = iterator.__next__()
    if status:
        print("Status OK!")
    try:
        for i, batch in enumerate(iterator):
            print(f"--- Received Batch #{i+1} with {len(batch)} samples ---")
            time.sleep(2)
    finally:
        iterator.close()

"""
