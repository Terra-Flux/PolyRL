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
import sys
import os
import asyncio
import logging
import time
import torch
import torch.multiprocessing as mp
from typing import Dict, List, Tuple, Optional
from aiohttp import web, ClientSession, ClientTimeout
import argparse
import requests

# utrans-engine should be installed via pip install -e .
from utrans_engine.configs.config import SenderConfig, ReceiverInfo
from utrans_engine.agents.sender import Sender
from utrans_engine.engine.utils import get_local_ip

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockRolloutManager:
    """Mock Rollout Manager to coordinate between SGLang and the Trainer."""
    def __init__(self, host: str, port: int, weight_sender_host: str, weight_sender_port: int):
        self.host = host
        self.port = port
        self.weight_sender_endpoint = f"{weight_sender_host}:{weight_sender_port}"
        self.registered_instances: Dict[str, Dict] = {}

    async def register_rollout_instance(self, request):
        data = await request.json()
        host = data.get('host')
        port = data.get('port')
        endpoint = f"{host}:{port}"
        logger.info(f"Registering rollout instance: {endpoint}")
        
        # Initialize instance state
        if endpoint not in self.registered_instances:
            self.registered_instances[endpoint] = {
                'host': host, 'port': port, 'ready': False, 'weight_version': 0
            }
        
        return web.json_response({
            'weight_sender_rpyc_endpoint': self.weight_sender_endpoint,
            'sender_group_idx': 0, 
            'num_mooncake_engines_per_group': 1
        })

    async def get_receive_instances(self, request):
        """Called by Sender Agent to find workers needing updates."""
        data = await request.json()
        sender_version = data.get('sender_weight_version', 0)
        
        instances = []
        for endpoint, d in self.registered_instances.items():
            if d['weight_version'] < sender_version:
                instances.append({
                    "instance": {"endpoint": f"http://{endpoint}"},
                    "current_weight_version": d['weight_version']
                })
        return web.json_response({"instances": instances})

    async def update_weights(self, request):
        """Called by Sender Agent to trigger the Sidecar application."""
        data = await request.json()
        endpoints = data.get('instance_endpoints', [])
        version = data.get('weight_version', 0)
        bootstrap = data.get('bootstrap', False)
        
        logger.info(f"Manager: Notifying {len(endpoints)} endpoints (v{version}, bootstrap={bootstrap})")
        
        loop = asyncio.get_event_loop()
        for endpoint in endpoints:
            # The Standalone Sidecar listens on its own handshake port
            sidecar_url = f"http://{endpoint}/update_weights_from_agent"
            payload = {
                "tensors_meta": data.get('tensors_meta'),
                "weight_version": version,
                "bootstrap": bootstrap
            }
            try:
                # Increased timeout for bootstrapping/sharding
                resp = await loop.run_in_executor(None, lambda: requests.post(sidecar_url, json=payload, timeout=10))
                if resp.status_code == 200:
                    if not bootstrap:
                        if endpoint in self.registered_instances:
                            self.registered_instances[endpoint]['weight_version'] = version
                    logger.info(f"Successfully notified sidecar at {endpoint}")
            except Exception as e:
                logger.error(f"Failed to notify sidecar {endpoint}: {e}")
                
        return web.json_response({"success": True})

    async def health_check_task(self):
        """Background task to poll health of registered instances."""
        while True:
            for endpoint in list(self.registered_instances.keys()):
                if not self.registered_instances[endpoint]['ready']:
                    try:
                        async with ClientSession(timeout=ClientTimeout(total=1)) as session:
                            async with session.get(f"http://{endpoint}/health") as resp:
                                if resp.status == 200:
                                    self.registered_instances[endpoint]['ready'] = True
                                    logger.info(f"Instance {endpoint} is now READY")
                    except:
                        pass
            await asyncio.sleep(2)

    def create_app(self):
        app = web.Application()
        app.router.add_post('/register_rollout_instance', self.register_rollout_instance)
        app.router.add_post('/get_receive_instances', self.get_receive_instances)
        app.router.add_post('/update_weights', self.update_weights)
        return app

class DummyTrainer:
    def __init__(self, model_path: Optional[str], sender_port: int):
        self.model_path = model_path
        self.sender_port = sender_port
        self.in_q = mp.Queue()
        self.out_q = mp.Queue()
        self.agent_proc = None
        self.buffer = None

    def generate_initial_weights(self):
        if self.model_path:
            try:
                from transformers import AutoModelForCausalLM
                logger.info(f"Loading weights from {self.model_path}...")
                # Load to CPU to avoid OOM on trainer side, no device_map to avoid accelerate dependency
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                return model.state_dict()
            except Exception as e:
                logger.warning(f"Failed to load real model: {e}. Falling back to dummy.")

        logger.info("Generating dummy weights structure...")
        return {
            "model.embed_tokens.weight": torch.randn(1024, 512, dtype=torch.bfloat16),
            "lm_head.weight": torch.randn(1024, 512, dtype=torch.bfloat16),
        }

    def start_agent(self, state_dict, manager_url: str):
        meta_sizes = []
        tensors_meta = []
        for name, param in state_dict.items():
            size = param.numel() * param.element_size()
            meta_sizes.append((name, size))
            tensors_meta.append((name, (list(param.shape), str(param.dtype).split('.')[-1])))
        
        config = SenderConfig(
            local_hostname=get_local_ip(), 
            handshake_port=19001, 
            rpyc_bind_port=self.sender_port
        )
        self.in_q.put(meta_sizes)
        self.in_q.put(tensors_meta)
        
        self.agent_proc = mp.Process(target=self._run_agent_proc, args=(config, self.in_q, self.out_q, manager_url))
        self.agent_proc.start()
        
        shm_name, shm_size = self.out_q.get(timeout=60)
        from multiprocessing import shared_memory
        self.shm_obj = shared_memory.SharedMemory(name=shm_name)
        self.buffer = torch.frombuffer(self.shm_obj.buf, dtype=torch.uint8)
        logger.info(f"Trainer agent started and buffer attached to {shm_name}.")

    @staticmethod
    def _run_agent_proc(config, in_q, out_q, manager_url: Optional[str] = None):
        agent = Sender(config, in_q, out_q, manager_url=manager_url)
        agent.start()

    def update_weights_in_shm(self, state_dict):
        offset = 0
        for name, param in state_dict.items():
            size = param.numel() * param.element_size()
            self.buffer[offset : offset + size].copy_(
                param.data.contiguous().view(-1).view(torch.uint8), non_blocking=True
            )
            offset += size
        
        self.in_q.put("update_weights")
        logger.info("Weights pushed to shm, weighting for transfer engine")
        # NOTE: consider get before put, so that weight update can be async
        status = self.out_q.get(timeout=120)
        if status != "completed":
            raise RuntimeError(f"Weight transfer failed: {status}")

def perturb_and_update(trainer, state_dict):
    # if randomize the weights
    # with torch.no_grad():
    #     for name in state_dict:
    #         state_dict[name] *= 0.01
    #         break
    logger.info("pushing weights to shm")
    trainer.update_weights_in_shm(state_dict)

async def periodic_update_loop(trainer, manager, state_dict, interval):
    version = 0
    loop = asyncio.get_event_loop()
    while True:
        ready_endpoints = [e for e, d in manager.registered_instances.items() if d['ready']]
        if not ready_endpoints:
            logger.info("No healthy rollout sidecars. Waiting...")
            await asyncio.sleep(5)
            continue

        version += 1
        logger.info(f"--- Starting Update Cycle {version} ---")
        try:
            await loop.run_in_executor(None, perturb_and_update, trainer, state_dict)
            logger.info(f"Cycle {version} finished.")
        except Exception as e:
            logger.error(f"Cycle {version} failed: {e}")
            
        await asyncio.sleep(interval)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--manager-port", type=int, default=8000)
    parser.add_argument("--sender-port", type=int, default=18861)
    args = parser.parse_args()

    local_ip = get_local_ip()
    manager_url = f"http://{local_ip}:{args.manager_port}"
    
    trainer = DummyTrainer(args.model, args.sender_port)
    weights = trainer.generate_initial_weights()
    trainer.start_agent(weights, manager_url)
    
    manager = MockRolloutManager(local_ip, args.manager_port, local_ip, args.sender_port)
    app_runner = web.AppRunner(manager.create_app(), access_log=None)
    await app_runner.setup()
    await web.TCPSite(app_runner, local_ip, args.manager_port).start()
    
    logger.info(f"Mock Rollout Manager started at {manager_url}")
    asyncio.create_task(manager.health_check_task())

    try:
        await periodic_update_loop(trainer, manager, weights, args.interval)
    except asyncio.CancelledError:
        pass
    finally:
        await app_runner.cleanup()
        if trainer.agent_proc:
            trainer.agent_proc.terminate()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested.")
