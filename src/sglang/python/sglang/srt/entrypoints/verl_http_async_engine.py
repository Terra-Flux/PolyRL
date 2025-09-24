import base64
import copy
import dataclasses
import multiprocessing
import pickle
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import torch
from PIL.Image import Image
import torch.distributed as dist

from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, kill_process_tree

import aiohttp
import json

def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:

    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()

    base_url = server_args.url()
    timeout = 300.0  # Increased timeout to 5 minutes for downloading large models
    start_time = time.perf_counter()

    with requests.Session() as session:
        while time.perf_counter() - start_time < timeout:
            try:
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Authorization": f"Bearer {server_args.api_key}",
                }
                response = session.get(f"{base_url}/health_generate", headers=headers)
                if response.status_code == 200:
                    return p
            except requests.RequestException:
                pass

            if not p.is_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)

    p.terminate()
    raise TimeoutError("Server failed to start within the timeout period.")

# polyrl-dev
# Implement async version of server-based engine
class HttpServerAsyncEngineAdapter(EngineBase):
    """
    You can use this class to launch a server from a VerlEngine instance.
    We recommend using this class only you need to use http server.
    Otherwise, you can use Engine directly.
    """

    def __init__(self, **kwargs):
        self.server_args = ServerArgs(**kwargs)
        print(
            f"Launch HttpServerEngineAdapter at: {self.server_args.host}:{self.server_args.port}"
        )
        self.process = launch_server_process(self.server_args)
        # polyrl-dev
        self._need_reload = True
        self.timeout = kwargs.get("timeout", 3000)

    def _make_request(self, endpoint: str, payload: Optional[dict] = None, decode: bool = True):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"
        response = requests.post(url, json=payload or {})
        response.raise_for_status()
        if decode:
            response = response.json()
        return response
    
    async def _make_async_request(self, endpoint: str, payload: Optional[dict] = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, json=payload or {}) as response:
                response.raise_for_status()
                try:
                    response = await response.json()
                    return response
                except json.JSONDecodeError:
                    # print(f"Warning: Could not decode JSON")
                    return response
            
    async def _make_stream_request(self, endpoint: str, payload: Optional[dict] = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            Response iterator
        """
        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(url, json=payload or {}) as response:
                response.raise_for_status()
                async for resp_chunk in response.content:
                    resp_chunk = resp_chunk.decode('utf-8').strip()
                    if resp_chunk.startswith("data:"):
                        data_str = resp_chunk[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            # meta_info = data.get("meta_info", {})
                            # text_chunk = data.get("text", "")
                            yield data
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON from line: {data_str}")
                            continue

    async def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, torch.Tensor]],
        load_format: Optional[str] = None,
        flush_cache: bool = False,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """

        return await self._make_async_request(
            "update_weights_from_tensor",
            {
                "serialized_named_tensors": [
                    MultiprocessingSerializer.serialize(named_tensors, output_str=True)
                    for _ in range(self.server_args.tp_size)
                ],
                "load_format": load_format,
                "flush_cache": flush_cache,
            },
        )

    def shutdown(self):
        kill_process_tree(self.process.pid)

    def generate(
        self,
        prompt=None,
        sampling_params=None,
        input_ids=None,
        image_data=None,
        return_logprob=False,
        logprob_start_len=None,
        top_logprobs_num=None,
        token_ids_logprob=None,
        lora_path=None,
        custom_logit_processor=None,
    ):
        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
            "input_ids": input_ids,
            "image_data": image_data,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "token_ids_logprob": token_ids_logprob,
            "lora_path": lora_path,
            "custom_logit_processor": custom_logit_processor,
        }
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}

        return self._make_request("generate", payload)
    
    async def async_generate(self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[
            Union[
                List[List[Union[Image, str]]],
                List[Union[Image, str]],
                Union[Image, str],
            ]
        ] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
    ):
        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
            "input_ids": input_ids,
            "image_data": image_data,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "token_ids_logprob": token_ids_logprob,
            "lora_path": lora_path,
            "custom_logit_processor": custom_logit_processor,
        }
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}

        return await self._make_async_request("generate", payload)
    
    async def stream_generate(self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[Union[List[str], str]] = None,
        sampling_params: Optional[Union[List[Dict], Dict]] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[Union[List[List[int]], List[int]]] = None,
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[
            Union[
                List[List[Union[Image, str]]],
                List[Union[Image, str]],
                Union[Image, str],
            ]
        ] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
    ):
        """
        Example:
        
        stream = engine.stream_generate(...)
        try: 
            async for chunk in await stream:
                ...
        except:
            ...
        finally:
            stream.close()
        """
        payload = {
            "text": prompt,
            "sampling_params": sampling_params,
            "input_ids": input_ids,
            "image_data": image_data,
            "return_logprob": return_logprob,
            "logprob_start_len": logprob_start_len,
            "top_logprobs_num": top_logprobs_num,
            "token_ids_logprob": token_ids_logprob,
            "lora_path": lora_path,
            "custom_logit_processor": custom_logit_processor,
            "stream": True,
        }
        # Filter out None values
        payload = {k: v for k, v in payload.items() if v is not None}

        return self._make_stream_request("generate", payload)

    async def release_memory_occupation(self):
        return await self._make_async_request("release_memory_occupation", decode=False)

    async def resume_memory_occupation(self):
        if self._need_reload:
            await self.release_memory_occupation()
            self._need_reload = False
        return await self._make_async_request("resume_memory_occupation", decode=False)

    async def flush_cache(self):
        return await self._make_async_request("flush_cache", decode=False)
