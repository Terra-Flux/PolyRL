# GKE setup CLI

This directory provides a single Python CLI to create the GKE cluster and GPU worker pools from TOML configs under `config/`.

Prerequisites
- Python 3.9+ with the `toml` package: `pip install toml`
- gcloud SDK authenticated and configured

Config shape (excerpt)
```toml
[general]
project = "example-project"
region = "us-central1"
zone = "us-central1-a"
cluster_name = "example-cluster"

[default_pool]
machine_type = "e2-medium"
min_nodes = 1
max_nodes = 4

[[worker_pools]]
machine_type = "n1-standard-1"
provision = "spot" # or "standard"
accelerator = "type=nvidia-tesla-t4,count=1,gpu-driver-version=latest"
min_nodes = 0
max_nodes = 4
```

Basic usage
```bash
# Create cluster from config/example.toml
python cluster_setup/gke_setup.py --config example create-cluster

# Skip connecting kubectl context after creation
python cluster_setup/gke_setup.py --config example create-cluster --skip-connect

# Create ALL worker pools defined in [[worker_pools]] (index -1 means all)
python cluster_setup/gke_setup.py --config example create-worker-pool -1

# Create only the Nth worker pool (0-based index)
python cluster_setup/gke_setup.py --config example create-worker-pool 0
python cluster_setup/gke_setup.py --config example create-worker-pool 1

# Delete-before-create when updating an existing pool
python cluster_setup/gke_setup.py --config example create-worker-pool 0 --delete-first

# Override base name and autoscaling bounds (a "-N" suffix will be added)
python cluster_setup/gke_setup.py --config example create-worker-pool 0 \
  --pool-name custom-pool --min-nodes 0 --max-nodes 6

# Dry-run to preview the gcloud commands without execution
python cluster_setup/gke_setup.py --dry-run --config example create-worker-pool -1

# Use an absolute TOML path
python cluster_setup/gke_setup.py --config /abs/path/to/file.toml create-cluster
```

Notes
- `--config` accepts a short name (resolved to `config/<name>.toml`) or an absolute path.
- Worker pool names default to `<machine_type>-<provision>-gpu-pool-N` (e.g., `a3-highgpu-2g-spot-gpu-pool-0`).
- Output is colorized where supported and shows a labeled "Command:" preview with line continuations.
- Set `NO_COLOR=1` to disable colored output.

