from typing import List, Optional, Tuple
import warnings
import requests
import torch
from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.server_args import ServerArgs

"""
This is a simple HTTP engine that can be used to interact with the server.
It assumes the server process is already running, specified in the ServerArgs.
It is used in the VerlEngine when the backend is "http".
"""

class HttpEngineForRL(EngineBase):
    # same as HttpServerEngineForRL but doesn't launch a server process
    # assume the server process is already running
    def __init__(self, **kwargs):
        self.server_args = ServerArgs(**kwargs)
        print(f"Connecting to server at {self.server_args.host}:{self.server_args.port}")
        # self.process = launch_server_process(self.server_args)
        model_info = self._query_model_info()
        print(f"Model info: {model_info}")
        self.verified_server_info = self._verify_server_info()
        # set them to None to avoid using them
        self.tokenizer_manager = None
        self.scheduler_info = None

    def _make_post_request(self, endpoint: str, payload: dict = None):
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
        return response.json()

    def _make_get_request(self, endpoint: str):
        """Make a GET request to the specified endpoint.

        Args:
            endpoint: The API endpoint to call
        """
        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def _query_model_info(self):
        """Query the model info from the server."""
        return self._make_get_request("get_model_info")

    def _verify_server_info(self):
        """Verify the server info is aligned with the server_args."""
        server_info = self._make_get_request("get_server_info")

        # check model_path
        if server_info["model_path"] != self.server_args.model_path:
            raise ValueError(f"Model path mismatch: {server_info['model_path']} != {self.server_args.model_path}")

        # check tokenizer_path
        if server_info["tokenizer_path"] != self.server_args.tokenizer_path:
            raise ValueError(f"Tokenizer path mismatch: {server_info['tokenizer_path']} != {self.server_args.tokenizer_path}")

        # check tp_size
        # if server_info["tp_size"] != self.server_args.tp_size:
        #     raise ValueError(f"TP size mismatch: {server_info['tp_size']} != {self.server_args.tp_size}")
        # NOTE: TP size is provided by the remote server
        print(f"TP size of server {self.server_args.host}:{self.server_args.port}: {server_info['tp_size']}")
        self.local_tp_size = self.server_args.tp_size
        self.server_args.tp_size = server_info["tp_size"]
        if self.local_tp_size % self.server_args.tp_size != 0:
            warnings.warn(f"TP size mismatch: local tp size {self.local_tp_size} is not divisible by remote tp size {self.server_args.tp_size}, which may cause errors in weight update.")

        # TODO: check other fields if necessary
            
        return server_info

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
        raise NotImplementedError("update_weights_from_tensor is not implemented for HttpEngineForRL")

    # polyrl-dev
    def update_weights_from_agent(
        self,
        tensors_meta: List[Tuple[str, Tuple[List[int], str]]],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        bootstrap: bool = False,
    ):
        # NOTE(yongji): 
        # Now each tp rank needs to accept a full weight
        # They will get the same tensor meta
        # In the current implementation of SGLang's update_weights_from_tensor, it is the same
        # But it copies the tensor meta tp_size times for _ in range(self.server_args.tp_size)
        return self._make_post_request(
            "update_weights_from_agent",
            {
                "tensors_meta": tensors_meta,
                "load_format": load_format,
                "flush_cache": flush_cache,
                "bootstrap": bootstrap,
            },
        )

    def shutdown(self):
        # kill_process_tree(self.process.pid)
        print("shutdown of HttpEngineForRL")

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
        print(f"HttpEngineForRL generating...")
        return self._make_post_request("generate", payload)

    def release_memory_occupation(self):
        print(f"HttpEngineForRL releasing memory occupation...")
        # TODO: enable this after update model weights is implemented
        # return self._make_post_request("release_memory_occupation")
        return None
    
    def resume_memory_occupation(self):
        print(f"HttpEngineForRL resuming memory occupation...")
        # TODO: enable this after update model weights is implemented
        # return self._make_post_request("resume_memory_occupation")
        return None

    def flush_cache(self):
        print(f"HttpEngineForRL flushing cache...")
        # FIXME: return from flush_cache is not json
        return self._make_post_request("flush_cache")

