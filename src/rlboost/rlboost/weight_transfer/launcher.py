import os

# polyrl-dev
import subprocess
import requests

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import (NodeAffinitySchedulingStrategy,
                                            PlacementGroupSchedulingStrategy)

from .utils import get_node_ips, filter_ips_by_config

def spawn_rollout_manager(config):
    """Spawn the rollout manager process and register weight senders."""
    # Only spawn for sglang-disaggregated rollout
    if config.actor_rollout_ref.rollout.name != "sglang-disaggregated":
        return None
        
    if not config.actor_rollout_ref.rollout.get("rollout_manager", {}).get("endpoint"):
        return None
        
    # Extract rollout manager config
    rollout_mgr_config = config.actor_rollout_ref.rollout.rollout_manager
    
    # Get the directory of current file and calculate relative path to rollout-manager
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    rollout_manager_dir = os.path.join(current_file_dir, "../../rollout-manager")
    rollout_manager_dir = os.path.abspath(rollout_manager_dir)
    
    # Build command line arguments for rollout manager
    cmd = ["cargo", "run", "--release", "--"]
    
    # Add config file if it exists
    if rollout_mgr_config.config_path is not None:
        config_file_path = rollout_mgr_config.config_path
    else:
        config_file_path = os.path.join(rollout_manager_dir, "config.toml")
        rollout_mgr_config.config_path = config_file_path
    if os.path.exists(config_file_path):
        cmd.extend(["--config-file", config_file_path])
    
    # Parse the endpoint to get bind address
    port = rollout_mgr_config.port
    cmd.extend(["--bind-addr", f"0.0.0.0:{port}"])
    
    # Start the rollout manager process with cargo run in the rollout-manager directory
    print(f"[Training] Starting rollout manager with command: {' '.join(cmd)}")
    process = subprocess.Popen(cmd, cwd=rollout_manager_dir)

    return process

# polyrl-dev
# TODO(liuxs): avoid assuming ray cluster when collecting sender ips, try to let each sender register individually
def prepare_weight_sender_ips(pg: PlacementGroup) -> str:
    specs = ray._private.state.state.placement_group_table(pg.id)
    node_id = specs["bundles_to_node_id"][0]
    worker_on_node = ray.remote(get_node_ips).options(
        num_cpus=0,
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
    ).remote()
    all_node_ips = ray.get(worker_on_node)
    return ','.join(all_node_ips)

def register_weight_senders(config, resource_pool_manager):
    """Register weight sender IPs based on resource pool specification."""
    if not resource_pool_manager or not resource_pool_manager.resource_pool_dict:
        raise RuntimeError("[Training] No resource pools available for weight sender registration")
        
    rollout_mgr_config = config.actor_rollout_ref.rollout.rollout_manager
    
    # Gather list-of-list of IPs per node
    weight_sender_ips_list = []
    
    for pool_name, resource_pool in resource_pool_manager.resource_pool_dict.items():
        assert pool_name == "global_pool"
        for pg in resource_pool.pgs:
            weight_sender_ips_str = prepare_weight_sender_ips(pg)
            per_node_ips = []
            # Split the comma-separated IPs and add unique ones (per node)
            for ip in weight_sender_ips_str.split(','):
                ip = ip.strip()
                if ip and ip not in per_node_ips:
                    per_node_ips.append(ip)
            weight_sender_ips_list.append(per_node_ips)
    
    if not weight_sender_ips_list:
        raise RuntimeError("[Training] No weight sender IPs found")
    
    print(f"[Training] Gather all available weight sender IPs: {weight_sender_ips_list}")
    
    # Update weight senders via REST API
    try:
        response = requests.put(
            f"{rollout_mgr_config.endpoint}/update_weight_senders",
            json={
                "weight_sender_ips": weight_sender_ips_list,
            },
            timeout=10
        )
        if response.status_code == 200:
            print("[Training] Successfully registered weight sender IPs")
        else:
            raise RuntimeError(f"[Training] Failed to register weight senders: {response.text}")
    except Exception as e:
        raise RuntimeError(f"[Training] Error registering weight senders: {e}")