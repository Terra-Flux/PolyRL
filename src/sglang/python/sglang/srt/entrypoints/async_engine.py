import base64
import copy
import multiprocessing
import pickle
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import requests
import torch
import torch.distributed as dist

from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, kill_process_tree

from sglang.srt.coordinator.rollout_iterator import RolloutIterator, RolloutConfig

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
This class is based on the HttpServerEngineAdapter class.
Besides the local server process, it also connects to multiple remote server processes.
By setting the size of "loading_buffer", it will adjust the number of prompts processed locally and remotely.

NOTE:
- For simplicity, the size of `loading_buffer` must be divisible by the chunk size (micro batch size).

The workflow is as follows:
1. Initialize the local server process.
2. Connect to the remote server processes and verify the server status (model info and tp size).
3. Initialize the RolloutIterator with the remote server addresses.
4. When `generate` is called
    - Split the first "loading buffer" size of prompts into local server.
    - Submit the remaining prompts to the RolloutIterator.
5. When loading buffer is done, push those results into the RolloutIterator.
6. The `generate` will return the RolloutIterator. It will be used as a general iterable type object (e.g. list).
6. The remote server will continue to process the prompts while the local hardware start training process.

"""

# polyrl-dev
# NOTE(yongji): not currently used
@dataclass
class RemoteServerConfig:
    address: str
    model_path: str
    tokenizer_path: str
    tp_size: int

def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process:

    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()

    base_url = server_args.url()
    timeout = 300.0  # Increased timeout to 5 minutes for downloading large models
    start_time = time.time()

    with requests.Session() as session:
        while time.time() - start_time < timeout:
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


class HttpServerEngineAsync(EngineBase):
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

        # set them to None to avoid using them
        self.tokenizer_manager = None
        self.scheduler_info = None
        self.rollout_config = None # initialize in `generate`
        self.rollout_iterator = None # initialize in `generate`
        self.remote_rollout_servers = []

    def _make_get_request(self, endpoint: str):
        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def _make_post_request(self, endpoint: str, 
                           payload: Optional[dict] = None,
                           address: Optional[str] = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if address is None:
            url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"
        else:
            url = f"http://{address}/{endpoint}"
        response = requests.post(url, json=payload or {})
        response.raise_for_status()
        return response.json()

    def _make_get_request(self, endpoint: str, address: Optional[str] = None):
        """Make a GET request to the specified endpoint.

        Args:
            endpoint: The API endpoint to call

        Returns:
            The JSON response from the server
        """
        if address is None:
            url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"
        else:
            url = f"http://{address}/{endpoint}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def _verify_server_info(self, address: str):
        """Verify the server info is aligned with the server_args."""
        server_info = self._make_get_request("get_server_info", address)

        # NOTE: model path and tokenizer path should align with the local server

        # check model_path
        if server_info["model_path"] != self.server_args.model_path:
            raise ValueError(f"Model path mismatch: {server_info['model_path']} != {self.server_args.model_path}")

        # check tokenizer_path
        if server_info["tokenizer_path"] != self.server_args.tokenizer_path:
            raise ValueError(f"Tokenizer path mismatch: {server_info['tokenizer_path']} != {self.server_args.tokenizer_path}")

        # check tp_size
        # TODO: support flexible tp size
        logger.debug(f"TP size of remote server {address}: {server_info['tp_size']}")
        if self.server_args.tp_size % server_info["tp_size"] != 0:
            raise ValueError(f"TP size mismatch: local tp size {self.server_args.tp_size} is not divisible by remote tp size {server_info['tp_size']}, which will cause errors in weight update.")

        # TODO: check other fields if necessary
            
        return RemoteServerConfig(
            address=address,
            model_path=server_info["model_path"],
            tokenizer_path=server_info["tokenizer_path"],
            tp_size=server_info["tp_size"],
        )

    def add_remote_servers(self, addresses: List[str]):
        if len(self.remote_rollout_servers) != 0:
            raise ValueError("Remote servers are already added.")
        # TODO: support dynamic adding and removing remote servers

        for address in addresses:
            try:
                server_info = self._make_get_request("get_server_info", address)
                self.remote_rollout_servers.append(self._verify_server_info(address, server_info))
            except Exception as e:
                logger.error(f"Failed to verify server info for {address}: {e}, skipping...")

        logger.info(f"Added {len(self.remote_rollout_servers)} remote rollout servers:\n{'\n'.join([f'  {server.address}' for server in self.remote_rollout_servers])}")

    def update_weights_from_tensor(
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

        return self._make_request(
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
        if self.rollout_iterator is not None:
            self.rollout_iterator.close()

    def generate_async(
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
        micro_batch_size=1,
        loading_buffer_size=0, # no loading buffer by default
    ):
        if len(self.remote_rollout_servers) == 0:
            raise ValueError("No available remote rollout servers. Please use `generate` instead.")
        n_samples = sampling_params.get("n", 1)
        out_of_order = False # TODO: determine by ppo/grpo
        # init rollout config and iterator
        # TODO: directly attach all the generate args to the rollout config
        self.rollout_config = RolloutConfig(
            hosts=[server.address for server in self.remote_rollout_servers],
            max_tokens=sampling_params.get("max_tokens", 1000),
            temperature=sampling_params.get("temperature", 0.9),
            n_samples=n_samples,
            return_tokens=return_logprob,
            out_of_order=out_of_order,
            chunk_size=micro_batch_size,
        )
        self.rollout_iterator = RolloutIterator(self.rollout_config)
        self.rollout_iterator.start_rollout_manager()
        # start processing immediately
        self.rollout_iterator.send_prompts(prompt[loading_buffer_size:])

        if loading_buffer_size > 0:
            loading_buffer = self.generate(
                prompt=prompt[:loading_buffer_size],
                sampling_params=sampling_params,
                input_ids=input_ids,
                image_data=image_data,
                return_logprob=return_logprob,
                logprob_start_len=logprob_start_len,
                top_logprobs_num=top_logprobs_num,
                token_ids_logprob=token_ids_logprob,
                lora_path=lora_path,
                custom_logit_processor=custom_logit_processor,
            )
        if out_of_order:
            self.rollout_iterator.add_local_results(loading_buffer)
        else:
            # chunk by n_samples
            assert len(loading_buffer) % n_samples == 0, f"local rollout engine returns wrong number of samples, {len(loading_buffer)} is not divisible by {n_samples}"
            self.rollout_iterator.add_local_results([
                loading_buffer[i:i+n_samples]
                for i in range(0, len(loading_buffer), n_samples)
            ])

        return self.rollout_iterator

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

        return self._make_post_request("generate", payload)

    def release_memory_occupation(self):
        return self._make_post_request("release_memory_occupation")

    def resume_memory_occupation(self):
        return self._make_post_request("resume_memory_occupation")
