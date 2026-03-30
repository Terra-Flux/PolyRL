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
import socket
import ipaddress
import pickle
import pybase64
import io
from typing import List, Any

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj: Any, output_str: bool = False) -> Any:
        from multiprocessing.reduction import ForkingPickler
        buf = io.BytesIO()
        # ForkingPickler doesn't take protocol in __init__
        pickler = ForkingPickler(buf, pickle.HIGHEST_PROTOCOL)
        pickler.dump(obj)
        data = buf.getvalue()
        if output_str:
            return pybase64.b64encode(data).decode("utf-8")
        return data

    @staticmethod
    def deserialize(data: Any) -> Any:
        from multiprocessing.reduction import ForkingPickler
        if isinstance(data, str):
            data = pybase64.b64decode(data)
        return pickle.loads(data)

def get_node_ips():
    """Detect the local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def get_node_ips():
    import psutil
    try:
        all_interfaces = psutil.net_if_addrs()
        ips = []
        for interface, addrs in all_interfaces.items():
            if interface != 'lo':
                for addr in addrs:
                    if addr.family == socket.AF_INET:
                        ips.append(addr.address)
        return ips
    except:
        return [get_local_ip()]
