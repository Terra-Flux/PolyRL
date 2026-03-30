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
import logging
import argparse
import multiprocessing as mp
import requests
import time

from utrans_engine.agents.receiver_gpu import GPUReceiver
from utrans_engine.configs.config import ReceiverConfig
from utrans_engine.engine.utils import get_local_ip

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StandaloneGPUReceiver")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sglang-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--manager-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--handshake-port", type=int, default=21000)
    args = parser.parse_args()

    config = ReceiverConfig(
        local_hostname=get_local_ip(),
        handshake_port=args.handshake_port,
        sglang_url=args.sglang_url,
        tp_size=args.tp_size,
        rollout_manager_url=args.manager_url
    )

    # These queues are unused in standalone mode but required by the base class
    in_q = mp.Queue()
    out_q = mp.Queue()

    agent = GPUReceiver(config, in_q, out_q)
    
    # Check SGLang health before starting
    logger.info("Waiting for SGLang...")
    while True:
        try:
            resp = requests.get(f"{args.sglang_url.rstrip('/')}/health", timeout=5)
            if resp.status_code == 200:
                break
        except:
            pass
        time.sleep(2)

    logger.info(f"Starting GPU Receiver agent on port {args.handshake_port}...")
    # BaseReceiver.start() handles its own event loop via asyncio.run()
    agent.start()

if __name__ == "__main__":
    main()
