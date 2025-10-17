# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
# polyrl-dev
import subprocess
import requests

from verl.trainer.ppo.stream_ray_trainer import StreamRayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type


def spawn_rollout_manager(config):
    """Spawn the rollout manager process and register weight senders."""
    # Only spawn for sglang-disaggregated rollout
    if config.actor_rollout_ref.rollout.name != "sglang-disaggregated":
        return None
        
    if not config.actor_rollout_ref.rollout.get("rollout_manager", {}).get("endpoint"):
        return None
        
    # Extract rollout manager config
    rollout_mgr_config = config.actor_rollout_ref.rollout.rollout_manager
    weight_sender_config = config.actor_rollout_ref.rollout.weight_sender
    
    # Get the directory of current file and calculate relative path to rollout-manager
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    rollout_manager_dir = os.path.join(current_file_dir, "../../../rollout-manager")
    rollout_manager_dir = os.path.abspath(rollout_manager_dir)
    
    # Build command line arguments for rollout manager
    cmd = ["cargo", "run", "--release", "--"]
    
    # Add mooncake transfer protocol if specified
    if weight_sender_config.transfer_protocol is not None:
        cmd.extend(["--mooncake-transfer-protocol", weight_sender_config.transfer_protocol])
    
    # Add mooncake transfer device name if specified  
    if weight_sender_config.transfer_device_name is not None:
        cmd.extend(["--mooncake-transfer-device-name", weight_sender_config.transfer_device_name])
    
    # Parse the endpoint to get bind address
    port = rollout_mgr_config.port
    cmd.extend(["--bind-addr", f"0.0.0.0:{port}"])
    
    # Start the rollout manager process with cargo run in the rollout-manager directory
    print(f"[Training] Starting rollout manager with command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, cwd=rollout_manager_dir)

    return process


def register_weight_senders(config, resource_pool_manager):
    """Register weight sender endpoints based on resource pool specification."""
    if not resource_pool_manager or not resource_pool_manager.resource_pool_dict:
        raise RuntimeError("[Training] No resource pools available for weight sender registration")
        
    rollout_mgr_config = config.actor_rollout_ref.rollout.rollout_manager
    weight_sender_config = config.actor_rollout_ref.rollout.weight_sender
    
    # Get weight sender endpoints from WEIGHT_SENDER_IP environment variables on each node
    weight_sender_endpoints = []
    num_mooncake_groups = weight_sender_config.get('num_mooncake_groups', 1)
    rpyc_base_port = weight_sender_config.get('rpyc_bind_base_port', 18861)
    
    for pool_name, resource_pool in resource_pool_manager.resource_pool_dict.items():
        assert pool_name == "global_pool"
        for pg in resource_pool.pgs:
            # Get WEIGHT_SENDER_IP from the placement group node
            from verl.single_controller.ray.base import prepare_weight_sender_ips
            allowed_sender_ips = weight_sender_config.get('allowed_sender_ips', '0.0.0.0/0')
            weight_sender_ips_str = prepare_weight_sender_ips(pg, allowed_sender_ips, num_mooncake_groups)
            first_ip = weight_sender_ips_str.split(',')[0]
            endpoint = f"{first_ip}:{rpyc_base_port}"
            if endpoint not in weight_sender_endpoints:
                weight_sender_endpoints.append(endpoint)
    
    if not weight_sender_endpoints:
        raise RuntimeError("[Training] No weight sender endpoints found")
    
    print(f"[Training] Registering weight sender endpoints: {weight_sender_endpoints}")
    
    # Update weight senders via REST API
    try:
        response = requests.put(
            f"{rollout_mgr_config.endpoint}/update_weight_senders",
            json={
                "weight_sender_rpyc_endpoints": weight_sender_endpoints,
                "num_mooncake_groups": num_mooncake_groups,
                "num_mooncake_engines_per_group": weight_sender_config.get('num_mooncake_engines_per_group', 1)
            },
            timeout=10
        )
        if response.status_code == 200:
            print("[Training] Successfully registered weight sender endpoints")
        else:
            raise RuntimeError(f"[Training] Failed to register weight senders: {response.text}")
    except Exception as e:
        raise RuntimeError(f"[Training] Error registering weight senders: {e}")


def get_custom_reward_fn(config):
    import importlib.util
    import sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}") from e

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        ray.init(
            runtime_env=get_ppo_ray_runtime_env(),
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()
    try:
        ray.get(runner.run.remote(config))
    finally:
        # Cleanup rollout manager if spawned
        if hasattr(runner, 'rollout_manager_process'):
            try:
                rollout_manager_process = ray.get(runner.get_rollout_manager_process.remote())
                if rollout_manager_process:
                    rollout_manager_process.terminate()
                    rollout_manager_process.wait()
                    print("[Training] Rollout manager process terminated")
            except Exception as e:
                print(f"[Training] Error terminating rollout manager: {e}")

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)

@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}
        # polyrl-dev
        self.rollout_manager_process = None

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            # from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
            from verl.workers.stream_fsdp_workers import StreamActorRolloutRefWorker
            
            # polyrl-dev
            # if using sglang-disagg, choose StreamActorRolloutRefWorker
            if config.actor_rollout_ref.rollout.name == "sglang-disaggregated":
                actor_rollout_cls = (
                    StreamActorRolloutRefWorker
                )
            else:
                raise ValueError("stream trainer must run with sglang-disaggregated")
            ray_worker_group_cls = RayWorkerGroup

        # elif config.actor_rollout_ref.actor.strategy == "megatron":
        #     from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

        #     actor_rollout_cls = (
        #         AsyncActorRolloutRefWorker
        #         if config.actor_rollout_ref.rollout.mode == "async"
        #         else ActorRolloutRefWorker
        #     )
        #     ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)

        return actor_rollout_cls, ray_worker_group_cls

    def add_critic_worker(self, config):
        """Add critic worker to role mapping."""
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                # polyrl
                # use stream critic worker for disagg rollout
                # FIXME: stream critic worker is not tested yet, use GRPO instead
                if config.actor_rollout_ref.rollout.name == "sglang-disaggregated":
                    from verl.workers.stream_fsdp_workers import StreamCriticWorker as CriticWorker
                else:
                    from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import CriticWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

        # elif config.critic.strategy == "megatron":
        #     from verl.workers.megatron_workers import CriticWorker

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role

        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        from verl.trainer.ppo.ray_trainer import Role

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            self.mapping[Role.RewardModel] = "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        """Add reference policy worker if KL loss or KL reward is used."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    # polyrl-dev
    def get_rollout_manager_process(self):
        return self.rollout_manager_process
        
    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets.
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Set rollout manager endpoint dynamically based on head node IP
        if config.actor_rollout_ref.rollout.rollout_manager.port:
            try:
                from ray.util.state import list_nodes
                nodes = list_nodes()
                head_node = next((node for node in nodes if node.is_head_node), None)
                if head_node:
                    head_ip = head_node.node_ip
                else:
                    # Fallback to the original method
                    nodes = ray.nodes()
                    head_node = next(node for node in nodes if node.get('Resources', {}).get('object_store_memory'))
                    head_ip = head_node['NodeManagerAddress']
            except ImportError:
                # Fallback if Ray State API is not available
                nodes = ray.nodes()
                head_node = next(node for node in nodes if node.get('Resources', {}).get('object_store_memory'))
                head_ip = head_node['NodeManagerAddress']
            
            rollout_mgr_port = config.actor_rollout_ref.rollout.rollout_manager.port
            config.actor_rollout_ref.rollout.rollout_manager.endpoint = f"http://{head_ip}:{rollout_mgr_port}"
            print(f"[Training] Set rollout manager endpoint to: {config.actor_rollout_ref.rollout.rollout_manager.endpoint}")

        # Start the training process.
        is_stream = config.trainer.get("stream_fit", False)
        trainer = StreamRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )

        # polyrl-dev
        current_node_id = ray.get_runtime_context().get_node_id()
        # Check if current node is head node using Ray State API
        try:
            from ray.util.state import list_nodes
            nodes = list_nodes()
            head_node = next((node for node in nodes if node.is_head_node), None)
            is_head_node = head_node and head_node.node_id == current_node_id
        except ImportError:
            # Fallback to the original method
            nodes = ray.nodes()
            head_node = next(node for node in nodes if node.get('Resources', {}).get('object_store_memory'))
            is_head_node = current_node_id == head_node['NodeID']
        
        if is_head_node and config.actor_rollout_ref.rollout.name == "sglang-disaggregated":
            self.rollout_manager_process = spawn_rollout_manager(config)

        # Initialize the workers of the trainer.
        trainer.init_workers()

        if is_head_node and config.actor_rollout_ref.rollout.name == "sglang-disaggregated":
            register_weight_senders(config, resource_pool_manager)

        if is_stream:
            min_stream_batch_size = config.actor_rollout_ref.rollout.get("min_stream_batch_size", 0)
            if min_stream_batch_size == 0:
                raise ValueError(f"min_stream_batch_size must be specified to use stream mode.")
            elif not config.actor_rollout_ref.actor.use_dynamic_bsz:
                update_actor_micro_batch = config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * config.trainer.n_gpus_per_node
                assert min_stream_batch_size % update_actor_micro_batch == 0, \
                    f"{min_stream_batch_size=} must be dividable by {update_actor_micro_batch=}"
                assert min_stream_batch_size % config.actor_rollout_ref.rollout.n == 0, \
                    f"{min_stream_batch_size=} must be dividable by {config.actor_rollout_ref.rollout.n=}"
            else:
                print(f"Use dynamic microbatch size with max tokens per gpu = {config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu}")
            
            # if critic is used, check if critic's microbatch size is equal to rollout's
            # FIXME(liuxs): check dynamic batch size on ppo
            from verl.trainer.ppo.ray_trainer import AdvantageEstimator
            if config.algorithm.adv_estimator == AdvantageEstimator.GAE: # use critic model in GAE
                assert config.critic.ppo_micro_batch_size_per_gpu == config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, \
                    "critic's microbatch size must be equal to rollout's in stream mode"

            trainer.fit()
        else:
            raise ValueError("main_stream must use run in stream mode")


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    elif "datagen" in data_config and data_config.datagen.get("path", None) is not None and is_train:
        # If a data generation strategy is specified, use the DynamicGenDataset class
        from verl.utils.dataset.dynamicgen_dataset import DynamicGenDataset

        dataset_cls = DynamicGenDataset
        print("Using DynamicGenDataset for data generation.")

    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_type(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )

    # Use a sampler to facilitate checkpoint resumption.
    # If shuffling is enabled in the data configuration, create a random sampler.
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        # If shuffling is disabled, use a sequential sampler to iterate through the dataset in order.
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
