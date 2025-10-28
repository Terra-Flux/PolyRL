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
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _style(kind: str, text: str) -> str:
    if not _supports_color():
        if kind == "title":
            return f"== {text} =="
        if kind == "warn":
            return f"[!] {text}"
        return text
    styles = {
        "title": "\033[1;36m",  # bold cyan
        "info": "\033[1;32m",   # bold green
        "warn": "\033[1;33m",   # bold yellow
        "dim": "\033[2m",       # dim
        "reset": "\033[0m",
    }
    prefix = styles.get(kind, "")
    reset = styles["reset"]
    return f"{prefix}{text}{reset}"


def _print_section(title: str) -> None:
    bar = "=" * max(6, min(78, len(title) + 10))
    print("\n" + _style("title", title))
    print(_style("dim", bar))


def load_toml_bytes(path: Path) -> dict:
    import toml

    with path.open("r", encoding="utf-8") as f:
        return toml.load(f)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_config_path(config: str) -> Path:
    candidate = Path(config)
    if candidate.exists():
        return candidate
    # Treat as a short name under config/<name>.toml
    by_name = repo_root() / "config" / f"{config}.toml"
    if by_name.exists():
        return by_name
    raise FileNotFoundError(f"Config not found: {config}")


def get_worker_pools(cfg: dict) -> list:
    pools = cfg.get("worker_pools", [])
    if isinstance(pools, dict):
        return [pools]
    return list(pools)


def run(cmd: list[str], dry_run: bool) -> None:
    cmd_str = " ".join(shlex.quote(c) for c in cmd).replace(" --", " \\\n     --")
    print(_style("info", "Command:"))
    print("$", cmd_str)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def create_cluster(cfg: dict, dry_run: bool, skip_connect: bool) -> None:
    g = cfg["general"]
    dpool = cfg.get("default_pool", {})

    project = str(g["project"]) 
    region = str(g["region"]) 
    zone = str(g["zone"]) 
    cluster_name = str(g["cluster_name"]) 
    machine_type = str(dpool.get("node_type", "e2-medium"))
    min_nodes = str(dpool.get("min_nodes", 1))
    max_nodes = str(dpool.get("max_nodes", 4))

    _print_section(f"Create cluster '{cluster_name}' in {region} ({zone})")
    print(_style("dim", f"project={project} machine_type={machine_type} autoscaling=[{min_nodes},{max_nodes}]"))

    cmd = [
        "gcloud",
        "container",
        "clusters",
        "create",
        cluster_name,
        "--project",
        project,
        "--region",
        region,
        "--node-locations",
        zone,
        "--release-channel",
        "regular",
        "--machine-type",
        machine_type,
        "--num-nodes",
        "1",
        "--enable-autoscaling",
        "--min-nodes",
        min_nodes,
        "--max-nodes",
        max_nodes,
        "--enable-ip-alias",
        "--workload-pool",
        f"{project}.svc.id.goog",
    ]
    
    if g.get("vpc_net") and g.get("vpc_subnet"):
        vpc_net = str(g["vpc_net"]) 
        vpc_subnet = str(g["vpc_subnet"]) 
        cmd.extend([
            "--network",
            vpc_net,
            "--subnetwork",
            vpc_subnet,
        ])
        
    run(cmd, dry_run)
    if skip_connect:
        print(_style("warn", "Next step:"))
        print("$", f"gcloud container clusters get-credentials {cluster_name} --region {region} --project {project}")
    else:
        _print_section("Connect kubectl context")
        credentials_cmd = [
            "gcloud",
            "container",
            "clusters",
            "get-credentials",
            cluster_name,
            "--region",
            region,
            "--project",
            project,
        ]
        run(credentials_cmd, dry_run)


def create_rollout_pool(cfg: dict, dry_run: bool, delete_first: bool, pool_name: str | None = None,
                        min_nodes: int | None = None, max_nodes: int | None = None) -> None:
    g = cfg["general"]
    rpool = cfg["rollout_pool"]

    project = str(g["project"]) 
    region = str(g["region"]) 
    zone = str(g["zone"]) 
    cluster_name = str(g["cluster_name"]) 

    machine_type = str(rpool["node_type"]) 
    accelerator = str(rpool["accelerator"]) 
    min_nodes_val = str(min_nodes if min_nodes is not None else rpool.get("min_nodes", 0))
    max_nodes_val = str(max_nodes if max_nodes is not None else rpool.get("max_nodes", 4))
    
    provision = str(rpool["provision"]).lower()
    if provision != "standard" and provision != "spot":
        print(_style("warn", f"Unknown provision type: {provision}. Using 'spot' by default. Specify 'spot' or 'standard'.")) 

    pool = pool_name or f"{machine_type}-{provision}-gpu-pool"

    _print_section(f"Create node pool '{pool}'")
    print(_style("dim", f"cluster={cluster_name} region={region} machine_type={machine_type} accel=[{accelerator}] {provision} autoscaling=[{min_nodes_val},{max_nodes_val}]"))

    if delete_first:
        _print_section(f"Delete existing node pool '{pool}' if present")
        del_cmd = [
            "gcloud",
            "container",
            "node-pools",
            "delete",
            pool,
            "--cluster",
            cluster_name,
            "--project",
            project,
            "--region",
            region,
        ]
        run(del_cmd, dry_run)

    create_cmd = [
        "gcloud",
        "container",
        "node-pools",
        "create",
        pool,
        "--cluster",
        cluster_name,
        "--project",
        project,
        "--region",
        region,
        "--node-locations",
        zone,
        "--machine-type",
        machine_type,
        "--accelerator",
        accelerator,
        f"--{provision}",
        "--enable-autoscaling",
        "--num-nodes",
        min_nodes_val,
        "--min-nodes",
        min_nodes_val,
        "--max-nodes",
        max_nodes_val,
        "--node-taints",
        "nvidia.com/gpu=present:NoSchedule",
        "--enable-image-streaming",
    ]
    run(create_cmd, dry_run)


def create_worker_pool_entry(cfg: dict, pool_entry: dict, index: int, dry_run: bool, delete_first: bool,
                             pool_name_override: str | None = None,
                             min_nodes_override: int | None = None,
                             max_nodes_override: int | None = None) -> None:
    g = cfg["general"]

    project = str(g["project"]) 
    region = str(g["region"]) 
    zone = str(g["zone"]) 
    cluster_name = str(g["cluster_name"]) 

    machine_type = str(pool_entry["machine_type"]) 
    accelerator = str(pool_entry.get("accelerator", ""))
    provision = str(pool_entry.get("provision", "spot")).lower()
    if provision not in ("standard", "spot"):
        print(_style("warn", f"Unknown provision type: {provision}. Using 'spot'."))
        provision = "spot"

    min_nodes_val = str(min_nodes_override if min_nodes_override is not None else pool_entry.get("min_nodes", 0))
    max_nodes_val = str(max_nodes_override if max_nodes_override is not None else pool_entry.get("max_nodes", 4))

    base_name = pool_name_override or f"{machine_type}-{provision}-gpu-pool"
    pool = f"{base_name}-{index}"

    _print_section(f"Create worker pool '{pool}' (index {index})")
    print(_style("dim", f"cluster={cluster_name} region={region} machine_type={machine_type} accel=[{accelerator}] {provision} autoscaling=[{min_nodes_val},{max_nodes_val}]"))

    if delete_first:
        _print_section(f"Delete existing node pool '{pool}' if present")
        del_cmd = [
            "gcloud",
            "container",
            "node-pools",
            "delete",
            pool,
            "--cluster",
            cluster_name,
            "--project",
            project,
            "--region",
            region,
        ]
        run(del_cmd, dry_run)

    create_cmd = [
        "gcloud",
        "container",
        "node-pools",
        "create",
        pool,
        "--cluster",
        cluster_name,
        "--project",
        project,
        "--region",
        region,
        "--node-locations",
        zone,
        "--machine-type",
        machine_type,
    ]

    if accelerator:
        create_cmd.extend(["--accelerator", accelerator])

    create_cmd.extend([
        f"--{provision}",
        "--enable-autoscaling",
        "--num-nodes",
        min_nodes_val,
        "--min-nodes",
        min_nodes_val,
        "--max-nodes",
        max_nodes_val,
        "--node-taints",
        "nvidia.com/gpu=present:NoSchedule",
        "--enable-image-streaming",
    ])

    run(create_cmd, dry_run)


def create_worker_pools(cfg: dict, index: int, dry_run: bool, delete_first: bool,
                        pool_name: str | None, min_nodes: int | None, max_nodes: int | None) -> None:
    pools = get_worker_pools(cfg)
    if index == -1:
        for i, entry in enumerate(pools):
            create_worker_pool_entry(cfg, entry, i, dry_run, delete_first, pool_name, min_nodes, max_nodes)
    else:
        if index < 0 or index >= len(pools):
            raise IndexError(f"worker pool index out of range: {index} (have {len(pools)})")
        create_worker_pool_entry(cfg, pools[index], index, dry_run, delete_first, pool_name, min_nodes, max_nodes)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Manage GKE cluster and GPU node pools using a TOML config (config/*.toml).\n\n"
            "Examples:\n"
            "  ./cluster_setup/gke_setup.py --config example create-cluster\n"
            "  ./cluster_setup/gke_setup.py --config example create-worker-pool -1\n"
            "  ./cluster_setup/gke_setup.py --dry-run --config example create-worker-pool 0 --delete-first\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to TOML or short name (e.g., 'example' => config/example.toml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print gcloud commands without executing",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    # Create cluster
    p_cluster = sub.add_parser("create-cluster", help="Create the GKE cluster defined in [general] and [default_pool]")
    p_cluster.add_argument("--skip-connect", action="store_true", help="Skip connection to cluster after creation")

    # Create rollout GPU node pool
    p_workers = sub.add_parser(
        "create-worker-pool",
        help="Create worker pool(s) based on [[worker_pools]] entries in TOML. Pass index n, or -1 for all.",
    )
    p_workers.add_argument("n", type=int, help="Worker pool index (0-based). Use -1 to create all")
    p_workers.add_argument("--delete-first", action="store_true", help="Delete pool if it exists")
    p_workers.add_argument("--pool-name", help="Override base pool name (default from machine type and provisioning)")
    p_workers.add_argument("--min-nodes", type=int, help="Override min nodes")
    p_workers.add_argument("--max-nodes", type=int, help="Override max nodes")

    args = parser.parse_args(argv)

    cfg_path = resolve_config_path(args.config)
    cfg = load_toml_bytes(cfg_path)

    if args.command == "create-cluster":
        create_cluster(cfg, args.dry_run, args.skip_connect)
    elif args.command == "create-worker-pool":
        create_worker_pools(
            cfg,
            index=args.n,
            dry_run=args.dry_run,
            delete_first=args.delete_first,
            pool_name=args.pool_name,
            min_nodes=args.min_nodes,
            max_nodes=args.max_nodes,
        )
    else:
        parser.error(f"Unknown command: {args.command}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)

