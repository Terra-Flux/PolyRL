# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import subprocess
import time

import hydra
import ray
import requests

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager


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
    
    # Get unique node IPs from resource pools used by training workers
    node_ips = set()
    for pool_name, resource_pool in resource_pool_manager.resource_pool_dict.items():
        assert pool_name == "global_pool"
        for pg in resource_pool.pgs:
            # Get node IP for each placement group
            from verl.single_controller.ray.base import get_pg_ip
            print(f"resource_pool.pgs: {pg}")
            node_ip = get_pg_ip(pg)
            node_ips.add(node_ip)
    
    if not node_ips:
        raise RuntimeError("[Training] No node IPs found from resource pools")
    
    # Build weight sender endpoints
    rpyc_base_port = weight_sender_config.get('rpyc_bind_base_port', 18861)
    weight_sender_endpoints = [f"{ip}:{rpyc_base_port}" for ip in sorted(node_ips)]
    
    print(f"[Training] Registering weight sender endpoints: {weight_sender_endpoints}")
    
    # Update weight senders via REST API
    try:
        response = requests.put(
            f"{rollout_mgr_config.endpoint}/update_weight_senders",
            json={"weight_sender_rpyc_endpoints": weight_sender_endpoints},
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
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN",
             "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
        )

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


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def __init__(self):
        self.rollout_manager_process = None
        
    def get_rollout_manager_process(self):
        return self.rollout_manager_process
        
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get('use_shm', False))

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge
            if config.actor_rollout_ref.model.get('lora_rank', 0) > 0:
                if not is_version_ge(pkg='vllm', minver='0.7.3'):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
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

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
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

        trainer.init_workers()

        if is_head_node and config.actor_rollout_ref.rollout.name == "sglang-disaggregated":
            register_weight_senders(config, resource_pool_manager)

        # TODO: add a config to control the fit or stream_fit
        is_stream = config.trainer.get("stream_fit", False)
        if is_stream:
            # NOTE: make sure we are using sglang-http for rollout
            # assert config.actor_rollout_ref.rollou.name == "sglang-disaggregated", "stream_fit is only supported for sglang-disaggregated rollout"
            # NOTE: make sure microbatch size is equal for rollout and ref_policy (ppo_micro_batch_size_per_gpu and log_prob_micro_batch_size_per_gpu)
            assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu \
                == config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu, \
                "ppo_micro_batch_size_per_gpu and log_prob_micro_batch_size_per_gpu must be equal in stream mode"
            # if critic is used, also check if critic's microbatch size is equal to rollout's
            from verl.trainer.ppo.ray_trainer import AdvantageEstimator
            if config.algorithm.adv_estimator == AdvantageEstimator.GAE: # use critic model in GAE
                assert config.critic.ppo_micro_batch_size_per_gpu == config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, \
                    "critic's microbatch size must be equal to rollout's in stream mode"

            # NOTE: if running in GRPO mode, make sure microbatch size divisible by n
            if config.actor_rollout_ref.rollout.n > 1:
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu % config.actor_rollout_ref.rollout.n == 0, \
                    "ppo_micro_batch_size_per_gpu must be divisible by rollout.n in stream mode"
            trainer.stream_fit()
        else:
            trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

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

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
