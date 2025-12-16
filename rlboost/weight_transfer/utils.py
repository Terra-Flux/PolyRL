import ipaddress
from dataclasses import dataclass
from typing import List

@dataclass
class TransferEngineConfig:
    """Configuration for Mooncake transfer engine."""
    local_hostname: str
    handshake_port: int


def get_node_ips():
    import socket
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
        try:
            hostname = socket.gethostname()
            return [socket.gethostbyname(hostname)]
        except:
            return []

# polyrl-dev
def filter_ips_by_config(all_ips: List[str], allowed_ips_config: str) -> List[str]:
    if allowed_ips_config == "0.0.0.0/0":
        return all_ips
    
    allowed_patterns = [s.strip() for s in allowed_ips_config.split(',')]
    filtered_ips = []
    
    for ip in all_ips:
        try:
            ip_obj = ipaddress.ip_address(ip)
            for pattern in allowed_patterns:
                if '/' in pattern:
                    if ip_obj in ipaddress.ip_network(pattern, strict=False):
                        filtered_ips.append(ip)
                        break
                elif ip == pattern:
                    filtered_ips.append(ip)
                    break
        except:
            continue
    
    return filtered_ips