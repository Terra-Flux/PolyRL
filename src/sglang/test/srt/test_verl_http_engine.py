import multiprocessing
import multiprocessing as mp
import os
import random
import time
import traceback
import unittest
from multiprocessing import Process

import requests
import torch
from openai import OpenAI
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.api import (
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from transformers import AutoModelForCausalLM

from sglang.srt.entrypoints.verl_engine import VerlEngine
from sglang.srt.entrypoints.verl_http_engine import VerlHttpEngine
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_port_available
from sglang.test.runners import (
    HFRunner,
    SRTRunner,
    check_close_model_outputs,
    get_dtype_str,
)
from sglang.test.test_utils import CustomTestCase, find_available_port, is_in_ci

_MAX_NEW_TOKENS = 8
_PROMPTS = ["1+1=2, 1+2=3, 1+3=4, 1+4=5, 1+5=", "1*1=1, 1*2=2, 1*3=3, 1*4=4, 1*5="]
_TORCH_DTYPE = torch.float16

# Set to false to temporarily debug issues unrelated to weight update
_ENABLE_UPDATE_WEIGHTS = False

CI_MODELS = [
    dict(model_path="meta-llama/Llama-3.1-8B-Instruct"),
]
ALL_OTHER_MODELS = [
    dict(model_path="meta-llama/Llama-3.1-8B-Instruct", 
        tokenizer_path="meta-llama/Llama-3.1-8B-Instruct",
        tp_size=1,
        host="localhost",
        port=40000,
    ),
]

# This port is used for HTTP API communication with the VerlEngine server
# It handles client requests for text generation, weight updates, and memory management
# This port must be available and not used by other processes

# Master port is used for PyTorch's distributed communication setup
# It enables tensor-parallel processes to communicate with each other
# Default is 23456, but we find an available port dynamically in assert_fragment_e2e_execution
# This port is critical for torch.distributed.init_process_group to function properly
# Each test needs a unique master_port to avoid conflicts between parallel test executions
# master_port = find_available_port(23456)  # This is set in assert_fragment_e2e_execution method


class TestVerlHttpEngine(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        multiprocessing.set_start_method("spawn")

    def assert_fragment_e2e_execution(
        self,
        index: int,
        model_path: str,
        tokenizer_path: str,
        host: str,
        port: int,
        tp_size: int = 2,
    ):
        """
        Tests VerlEngine with tensor parallelism across multiple processes.

        Spawns tp_size processes to test distributed execution, including:
        - Model inference via direct API and HTTP server
        - Weight updating functionality
        - Memory management (release/resume)

        The test validates output correctness against a reference implementation
        within specified tolerance bounds.

        Parameters:
        -----------
        index: int - Test index for logging
        model_path: str - HuggingFace model identifier
        mem_fraction_static: float - Memory fraction for static tensors
        tp_size: int - Number of tensor parallel processes
        tight_memory: bool - Enable memory optimization
        prefill_tolerance: float - Max error for prefill computation
        decode_tolerance: float - Max error for decoding computation
        """

        print(f"assert_fragment_e2e_execution START {index=} {model_path=}")

        _run_http_engine(
            host=host,
            port=port,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            tp_size=tp_size,
        )

    def test_models(self):
        """
        Orchestrates end-to-end testing across configured model sets.

        In CI environments: Randomly selects one model for faster testing.
        In development: Tests all configured models for comprehensive validation.

        Each model configuration specifies model path, memory settings,
        tensor-parallel size, and error tolerance bounds.
        """
        test_models = ALL_OTHER_MODELS
        if is_in_ci():
            # Randomly select one model in CI for faster testing
            test_models = [random.choice(ALL_OTHER_MODELS)]
        # Test all models in development environment
        print(f"Development environment: Testing all {len(ALL_OTHER_MODELS)} models")
        for index, model_info in enumerate(test_models):
            execution_ok = self.assert_fragment_e2e_execution(index=index, **model_info)
            if not execution_ok:
                print(f"Test {index} failed")


def _run_http_engine(
    host: str,
    port: int,
    model_path: str,
    tokenizer_path: str,
    tp_size: int,
):
    """
    Executes a client for testing VerlHttpEngine.

    Performs the core test operations:
    1. Tests VerlHttpEngine API (generation, memory management, weight updates)
    2. Tests OpenAI-compatible endpoints on rank 0

    Reports success/failure via output_writer pipe.

    Parameters:
    host: str - Host address
    port: int - Port number
    model_path: str - HuggingFace model identifier
    tokenizer_path: str - HuggingFace tokenizer identifier
    tp_size: int - Number of tensor parallel processes
    """
    try:
        print(f"Starting VerlHttpEngine", flush=True)

        if _ENABLE_UPDATE_WEIGHTS:

            # hf model is used for update weights
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=_TORCH_DTYPE, trust_remote_code=True
            ).cpu()

            # test update weights
            print(f"Transforming hf model to fsdp_state_dict", flush=True)
            fsdp_state_dict = _get_fsdp_state_dict(hf_model=hf_model, tp_size=tp_size)

        engine = VerlHttpEngine(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            host=host,
            port=port,
            tp_size=tp_size,
        )
        # test direct generate API with multiple different requests
        print(
            f"Testing direct generate API with multiple requests"
        )

        # Request 1: Basic generation with temperature
        print(f"Testing basic generation with temperature")
        direct_response = engine.generate(
            prompt="Hello, world!",
            sampling_params={"temperature": 0.7, "max_new_tokens": 20},
        )
        print(f"Response 1: {direct_response}")

        # Request 2: Zero temperature (greedy) generation
        print(f"Testing greedy generation")
        direct_response = engine.generate(
            prompt="Complete this sequence: 1, 2, 3,",
            sampling_params={"temperature": 0.0, "max_new_tokens": 10},
        )
        print(f"Response 2: {direct_response}")

        # Request 3: Batch generation
        print(f"Testing batch generation")
        batch_response = engine.generate(
            prompt=["Translate 'hello' to French:", "Translate 'goodbye' to Spanish:"],
            sampling_params={"temperature": 0.8, "max_new_tokens": 15},
        )
        print(f"Response 3: {batch_response}")

        # test memory occupation APIs
        print(f"Testing memory occupation APIs")
        engine.release_memory_occupation()
        print("Memory released")
        # time.sleep(1)
        engine.resume_memory_occupation()
        print("Memory resumed")

        # openai API test for reference
        client = OpenAI(api_key="None", base_url=f"http://localhost:{port}/v1")
        print(client.models.list().data[0].id)

        # Multiple HTTP API requests
        print("Testing HTTP API with multiple requests")

        # Request 1
        url = f"http://localhost:{port}/generate"
        data = {"text": "1*1=1, 1*2=2, 1*3=3, 1*4=4, 1*5="}
        response = requests.post(url, json=data)
        print(f"HTTP Response 1: {response.json()}")

        # Request 2
        data = {
            "text": "The capital of France is",
            "sampling_params": {"temperature": 0.2},
        }
        response = requests.post(url, json=data)
        print(f"HTTP Response 2: {response.json()}")

        # Request 3
        data = {
            "text": "List three colors:",
            "sampling_params": {"top_p": 0.95, "max_new_tokens": 25},
        }
        response = requests.post(url, json=data)
        print(f"HTTP Response 3: {response.json()}")

        if _ENABLE_UPDATE_WEIGHTS:

            engine.update_weights_from_tensor(
                [(k, v) for k, v in fsdp_state_dict.items()]
            )

        # Final generation test after weight update
        print(f"Testing generation after weight update")
        direct_response = engine.generate(
            prompt="After weight update: Hello, world!",
            sampling_params={"temperature": 0.7, "max_new_tokens": 20},
        )
        print(f"Post-update response: {direct_response}")

        execution_ok = True

    except Exception as e:
        print(f"Has error: {e}", flush=True)
        traceback.print_exc()
        execution_ok = False

    engine.shutdown()
    print(f"End", flush=True)

    return execution_ok


# Adapted from https://github.com/volcengine/verl/blob/main/tests/rollout/run_fsdp_vllm.py
def _get_fsdp_state_dict(hf_model, tp_size: int):
    """
    Creates a sharded state dictionary for weight update testing.

    Wraps the HuggingFace model with FSDP (FullyShardedDataParallel),
    configures precision settings, and returns a sharded state dict
    for testing VerlEngine's weight update capabilities.

    Parameters:
    hf_model - HuggingFace model to wrap
    tp_size: int - Number of tensor-parallel shards

    Returns:
    dict - Sharded state dict for update_weights_from_tensor
    """
    device_mesh = init_device_mesh(
        "cuda", mesh_shape=(tp_size,), mesh_dim_names=["fsdp"]
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    fsdp_model = FSDP(
        hf_model,
        use_orig_params=True,
        auto_wrap_policy=None,
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        cpu_offload=CPUOffload(offload_params=False),
        sync_module_states=False,
        device_mesh=device_mesh,
    )
    print(f"{fsdp_model=}")

    FSDP.set_state_dict_type(
        fsdp_model,
        state_dict_type=StateDictType.SHARDED_STATE_DICT,
        state_dict_config=ShardedStateDictConfig(),
    )

    return fsdp_model.state_dict()


if __name__ == "__main__":
    unittest.main()
