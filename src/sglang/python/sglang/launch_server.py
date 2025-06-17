"""Launch the inference server."""

import os
import sys

# FIXME(yongji): make it configurable to choose between normal sglang server or our async server
# polyrl-dev(liuxs): use async server instead of sync server
# async server suports all functions of sync server
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
