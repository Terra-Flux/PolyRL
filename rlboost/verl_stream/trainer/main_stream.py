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
from rlboost.weight_transfer.launcher import spawn_rollout_manager, register_weight_senders
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

from rlboost.verl_stream.trainer.ppo.stream_ray_trainer import StreamRayPPOTrainer
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type

from rlboost.verl_stream.trainer.ppo.reward import load_reward_manager

@hydra.main(config_path="config", config_name="ppo_stream_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
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
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

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
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)

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
            from rlboost.verl_stream.workers.stream_fsdp_workers import StreamActorRolloutRefWorker
            
            # polyrl-dev
            # if using sglang-disagg, choose StreamActorRolloutRefWorker
            if config.actor_rollout_ref.rollout.name == "sglang-disaggregated":
                actor_rollout_cls = (
                    StreamActorRolloutRefWorker
                )
            else:
                raise ValueError("stream trainer must run with sglang-disaggregated")
            ray_worker_group_cls = RayWorkerGroup

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
                    from rlboost.verl_stream.workers.stream_fsdp_workers import StreamCriticWorker as CriticWorker
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
        # TODO Here you can use the new registration method to support dynamic registration of roles
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        self.mapping[Role.ActorRollout] = global_pool_id
        self.mapping[Role.Critic] = global_pool_id
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)
        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker if enabled."""
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable"]:
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    from verl.workers.fsdp_workers import RewardModelWorker
                elif config.reward_model.strategy == "megatron":
                    from verl.workers.megatron_workers import RewardModelWorker
                else:
                    raise NotImplementedError
            elif use_legacy_worker_impl == "disable":
                from verl.workers.roles import RewardModelWorker

                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            if config.reward_model.enable_resource_pool:
                self.mapping[Role.RewardModel] = "reward_pool"
            else:
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

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

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
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

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
        # Get head node ip and spawn rollout manager
        current_node_id = ray.get_runtime_context().get_node_id()
        # Check if current node is head node using Ray State API
        try:
            from ray.util.state import list_nodes
            nodes = list_nodes()
            head_node = next((node for node in nodes if node.is_head_node), None)
            is_head_node = head_node and head_node.node_id == current_node_id
            head_ip = head_node.node_ip
        except ImportError:
            # Fallback to the original method
            nodes = ray.nodes()
            head_node = next(node for node in nodes if node.get('Resources', {}).get('object_store_memory'))
            is_head_node = current_node_id == head_node['NodeID']
            head_ip = head_node['NodeManagerAddress']
        
        assert config.actor_rollout_ref.rollout.name == "sglang-disaggregated", "main stream runs under rollout.name == sglang-disaggregated"
        rollout_mgr_port = config.actor_rollout_ref.rollout.rollout_manager.port
        config.actor_rollout_ref.rollout.rollout_manager.endpoint = f"http://{head_ip}:{rollout_mgr_port}"
        print(f"[Training] Set rollout manager endpoint to: {config.actor_rollout_ref.rollout.rollout_manager.endpoint}")
        if is_head_node:
            self.rollout_manager_process = spawn_rollout_manager(config)

        # Initialize the workers of the trainer.
        trainer.init_workers()

        # polyrl-dev
        if is_head_node:
            register_weight_senders(config, resource_pool_manager)

        # polyrl-dev
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


if __name__ == "__main__":
    main()
