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
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 1. Register patches immediately via wrapt hooks
# This must happen before any sgl_jax modules are imported
import utrans_patch.autopatch 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    # 2. Launch the original SGLang-JAX server
    logger.info("Launching SGLang-JAX server with automatic UTrans hooks...")
    try:
        from sgl_jax.srt.server_args import ServerArgs
        from sgl_jax.srt.entrypoints import http_server
        from sgl_jax.srt.utils import kill_process_tree

        server_args = ServerArgs.from_cli()
        
        if server_args.multimodal:
            from sgl_jax.srt.multimodal.entrypoint import http_server as multimodal_http_server
            multimodal_http_server.launch(server_args)
        else:
            http_server.launch(server_args)
        
    except Exception as e:
        logger.error(f"SGLang-JAX server crash: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        try:
            kill_process_tree(os.getpid(), include_parent=False)
        except:
            pass

if __name__ == "__main__":
    main()
