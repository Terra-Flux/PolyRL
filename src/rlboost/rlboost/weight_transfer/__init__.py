"""
Weight transfer agent is now separated from veRL as a independent component

Workflow:
    1. main:spawn_rollout_manager (on header node)
    2. main:init_workers
    2. main:register_weight_senders (on header node, with all available ips) -> rollout_mgr:update_weight_senders (filter and repeat ips)
    3. StreamActorRolloutRefWorker:FSDPSGLangShardingManager:FSDPInterface (spawn weight_transfer)
    4. FSDPSGLangShardingManager.weight_transfer:update_weights_with_agent (update weight in the shared buffer)
"""
from .agent import start_transfer_agent
from .config import *
from .launcher import spawn_rollout_manager, register_weight_senders
from .fsdp_interface import FSDPInterface

